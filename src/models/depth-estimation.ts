import * as ort from "onnxruntime-web/all";
import * as cv from "@techstark/opencv-js";
import type { BaseModel } from "./base-model.ts";

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

        // local vars
        const img_src = cv.imread(input[0]);
        let img_dst = new cv.Mat();

        // populate cv mat
        cv.cvtColor(img_src, img_dst, cv.COLOR_RGBA2RGB);
        let d_img = cv.blobFromImage(
            img_dst,
            1/255,
            new cv.Size(width, height),
            new cv.Scalar(0.485, 0,456, 0.406), // FIXME: this is the mean value from imagenet. is this acceptable?
            false,
        );
        img_dst.delete();

        // create float32 array
        let img_flat = new Float32Array(3 * width * height);
        for (let i = 0; i < width * height; i++) {
            const base = i * 3;
            const ch_base = i; // spatial index per channel
            img_flat[0 * width * height + ch_base] = d_img.data[base]     / 255;
            img_flat[1 * width * height + ch_base] = d_img.data[base + 1] / 255;
            img_flat[2 * width * height + ch_base] = d_img.data[base + 2] / 255;
        }
        d_img.delete();

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

    public postprocess(input: ort.Tensor, width: number, height: number): HTMLCanvasElement {
        // extract float32 data from the ort tensor
        const data = input.data as Float32Array;
        const shape = input.dims;

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

    // input_image: (1, 4, W, H). mask: (1, 1, W, H). output: (1, 4, W, H) 
    public segment_into_layers(input_image: ort.Tensor, depth_map: ort.Tensor, width: number, height: number, num_layers: number): Array<ort.Tensor>{

        let layers: Array<ort.Tensor> = [];
        const band_size = 256 / num_layers;

        // FIXME: this operation is expensive. O(width * height * num_layers). This can be reduced to O(width * height)
        for (let layer = 0; layer < num_layers; layer++) {
            const layer_min_val = Math.floor(layer * band_size);
            const layer_max_val = (layer === num_layers - 1) ? 255 : Math.floor((layer + 1) * band_size) - 1; // if last layer, then upper bound is 255

            let layer_pixels: Float32Array = new Float32Array(4 * width * height);
            for (let i = 0; i < width * height; i++) {
                const j = i * 4;
                const depth = depth_map.data[i] as number;
                if (depth >= layer_min_val && depth <= layer_max_val) {
                    layer_pixels[0 * width * height + i] = input_image.data[0 * width * height + i] as number;
                    layer_pixels[1 * width * height + i] = input_image.data[1 * width * height + i] as number;
                    layer_pixels[2 * width * height + i] = input_image.data[2 * width * height + i] as number;
                    layer_pixels[3 * width * height + i] = input_image.data[3 * width * height + i] as number;
                } else { // TODO: else block is bad for GPUs (bc of exec_mask). will it be better if we initialize all arrays as 0.0?
                    layer_pixels[0 * width * height + i] = 0.0;
                    layer_pixels[1 * width * height + i] = 0.0;
                    layer_pixels[2 * width * height + i] = 0.0;
                    layer_pixels[3 * width * height + i] = 0.0;
                }
            }

            const ort_layer = new ort.Tensor("float32", layer_pixels, [1, 4, width, height])

            layers.push(ort_layer);
        }

        return layers;
    }
}