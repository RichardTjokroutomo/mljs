import * as ort from "onnxruntime-web";
import type { BaseModel } from "./base-model.ts";
import { threshold } from "@techstark/opencv-js";

export class DepthEstimation implements BaseModel {
    ort_session: ort.InferenceSession | null = null;

    public constructor() {}

    public async create_session(model_path: Uint8Array, options: ort.InferenceSession.SessionOptions): Promise<void>{
        this.ort_session = await ort.InferenceSession.create(model_path, options);
    }

    public preprocess(input: Array<HTMLCanvasElement>, width: number, height: number): Array<ort.Tensor>{
        if (input.length == 0){
            throw new Error("input array is empty, can't preprocess image for depth estimation!");
        }

        // resize
        const resized = document.createElement("canvas");
        resized.width = width;
        resized.height = height;
        const ctx = resized.getContext("2d")!;
        ctx.drawImage(input[0], 0, 0, width, height);

        // extract RGBA pixel data
        const image_data = ctx.getImageData(0, 0, width, height);
        const pixels = image_data.data;

        // create float32 array in CHW layout, normalize to [0, 1]
        const img_flat = new Float32Array(3 * width * height);
        for (let i = 0; i < width * height; i++) {
            const base = i * 4;
            img_flat[0 * width * height + i] = pixels[base]     / 255;
            img_flat[1 * width * height + i] = pixels[base + 1] / 255;
            img_flat[2 * width * height + i] = pixels[base + 2] / 255;
        }

        // create ort tensor & return
        return [new ort.Tensor("float32", img_flat, [1, 3, width, height])];
    }

    public async run_inference(input_tensors: Array<ort.Tensor>): Promise<ort.Tensor> {
        if (this.ort_session === null) {
            throw new Error("Depth Estimation model hasn't been initialized yet!");
        }

        if (input_tensors.length == 0) {
            throw new Error("input_tensors array is empty, can't perform depth estimation!");
        }
        const feeds = {"l_x_": input_tensors[0]}; // the name of input node of DA_v2.
        let result = await this.ort_session.run(feeds);
        return result["select_36"]; // the name of output node of DA_v2
    }

    public postprocess(input: Array<ort.Tensor>, width: number, height: number): HTMLCanvasElement {
        // extract float32 data from the ort tensor
        const data = input[0].data as Float32Array;
        const shape = input[0].dims;

        // determine spatial dimensions from shape (e.g., [1, 1, H, W] or [1, H, W])
        let out_h: number, out_w: number;
        if (shape.length === 4) {
            out_h = shape[2];
            out_w = shape[3];
        } else if (shape.length === 3) {
            out_h = shape[1];
            out_w = shape[2];
        } else {
            throw new Error(`Unexpected tensor shape: [${shape.join(", ")}]`);
        }

        // find min/max for normalization
        let min_val = Infinity;
        let max_val = -Infinity;
        for (let i = 0; i < data.length; i++) {
            const v = data[i];
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
        }
        const range = max_val - min_val || 1;

        // create an offscreen canvas
        const canvas = document.createElement("canvas");
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext("2d")!;

        // build rgba ImageData from normalized depth values
        const image_data = ctx.createImageData(width, height);
        const pixels = image_data.data;

        // scale depth map to target dimensions using nearest-neighbor sampling
        const scale_x = out_w / width;
        const scale_y = out_h / height;
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const src_x = Math.min(Math.floor(x * scale_x), out_w - 1);
                const src_y = Math.min(Math.floor(y * scale_y), out_h - 1);
                const src_idx = src_y * out_w + src_x;
                const depth = (data[src_idx] - min_val) / range;
                const val = Math.round(depth * 255);
                const dst_idx = (y * width + x) * 4;
                pixels[dst_idx]     = val;     // R
                pixels[dst_idx + 1] = val;     // G
                pixels[dst_idx + 2] = val;     // B
                pixels[dst_idx + 3] = 255;     // A
            }
        }

        ctx.putImageData(image_data, 0, 0);
        return canvas;
    }

    public segment_into_layers(input_image: HTMLCanvasElement, depth_map: HTMLCanvasElement, width: number, height: number, num_layers: number): Array<HTMLCanvasElement>{
        const input_ctx = input_image.getContext("2d")!;
        const input_data = input_ctx.getImageData(0, 0, width, height);
        const depth_ctx = depth_map.getContext("2d")!;
        const depth_data = depth_ctx.getImageData(0, 0, width, height);

        const layers = [];
        const band_size = 256 / num_layers;

        for (let layer = 0; layer < num_layers; layer++) {
            const minDepth = Math.floor(layer * band_size);
            const maxDepth = (layer === num_layers - 1) ? 255 : Math.floor((layer + 1) * band_size) - 1;

            const canvas = document.createElement("canvas");
            canvas.width = width;
            canvas.height = height;
            const ctx = canvas.getContext("2d")!;
            const imageData = ctx.createImageData(width, height);
            const pixels = imageData.data;

            for (let i: number = 0; i < width * height; i++) {
                const j = i * 4;
                const depth = depth_data.data[j]; // R channel of grayscale depth
                if (depth >= minDepth && depth <= maxDepth) {
                    pixels[j]     = input_data.data[j];
                    pixels[j + 1] = input_data.data[j + 1];
                    pixels[j + 2] = input_data.data[j + 2];
                    pixels[j + 3] = 255;
                } else {
                    pixels[j + 3] = 0; // transparent
                }
            }

            ctx.putImageData(imageData, 0, 0);

            layers.push(canvas);
        }

        return layers;
    }
}