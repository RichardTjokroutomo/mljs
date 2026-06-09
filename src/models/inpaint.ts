import type { BaseModel } from "./base-model.ts";
import cvModule from "@techstark/opencv-js";
import * as ort from "onnxruntime-web";
import { ort_tensor_to_html_canvas } from "../utils/type_converter.ts";
import { resize_html_canvas, resize_canvas_native } from "../utils/html_canvas_manipulator.ts";

const cv = (cvModule as any).default ?? cvModule; // Handle both default and named exports from OpenCV.js

export class Inpaint implements BaseModel {
    ort_session: ort.InferenceSession | null = null;

    public constructor() {};

    public async create_session(model_path: Uint8Array, options: ort.InferenceSession.SessionOptions): Promise<void> {
        this.ort_session = await ort.InferenceSession.create(model_path, options);
    }

    public preprocess(input: Array<HTMLCanvasElement>, width: number, height: number): Array<ort.Tensor> {
        if (input.length != 3){
            throw new Error("input array is not 3, can't preprocess image for inpainting!");
        }

        // 1. preprocess input image
        const processed_input_image = this.preprocess_image(input[0], width, height);

        // 2. preprocess mask
        const processed_mask = this.preprocess_mask(input[1], width, height);
        const processed_current_layer = this.preprocess_mask(input[2], width, height);
        // const inverted_mask = this.invert_tensor(processed_mask);

        // return!
        return [processed_input_image, processed_mask, processed_current_layer];
    }

    public async run_inference(input_tensors: Array<ort.Tensor>): Promise<ort.Tensor> {
        if (this.ort_session === null) {
            throw new Error("Inpaint model hasn't been initialized yet!");
        }

        if (input_tensors.length != 2) {
            throw new Error("input_tensors array's length is not 2! Can't perform inference!");
        }

        const feeds = { image: input_tensors[0], mask: input_tensors[1] }; // convention: input image is first element; mask is second.
        let result = await this.ort_session.run(feeds);

        return result["result"];
    }

    public postprocess(inputs: Array<ort.Tensor>, width: number, height: number): HTMLCanvasElement {
        // convert inputs[0] to canvas
        let input_canvas: HTMLCanvasElement = ort_tensor_to_html_canvas(inputs[0]);

        // combine mask 1 & 2
        const combined_mask: ort.Tensor = this.create_crop_mask(inputs[1], inputs[2]);

        // read pixel data from the inpainted result canvas (512x512)
        const ctx = input_canvas.getContext("2d")!;
        const imageData = ctx.getImageData(0, 0, width, height);
        const pixels = imageData.data;

        const mask_data = combined_mask.data; // float32 in [0, 1]

        for (let i = 0; i < width * height; i++) {
            const keep = mask_data[i] as number; // smooth transition at crop boundary
            const j = i * 4;
            pixels[j]     = Math.round(pixels[j]     * keep); // set value to 0 if the coord doesn't overlap with the combined mask
            pixels[j + 1] = Math.round(pixels[j + 1] * keep);
            pixels[j + 2] = Math.round(pixels[j + 2] * keep);
            pixels[j + 3] = Math.round(pixels[j + 3] * keep);
        }

        ctx.putImageData(imageData, 0, 0);
        return input_canvas;

    }

    public combine_inpainted_layer_with_original(inpainted_layer: HTMLCanvasElement, original_image: HTMLCanvasElement): HTMLCanvasElement {
        const inpainted_ctx = inpainted_layer.getContext("2d")!;
        const original_ctx = original_image.getContext("2d")!;

        const inpainted_data = inpainted_ctx.getImageData(0, 0, inpainted_layer.width, inpainted_layer.height);
        const original_data = original_ctx.getImageData(0, 0, original_image.width, original_image.height);

        const inpainted_pixels = inpainted_data.data;
        const original_pixels = original_data.data;

        for (let i: number = 0; i < inpainted_layer.width * inpainted_layer.height; i++) {
            const j = i * 4;
            if (inpainted_pixels[j + 3] > 0  && original_pixels[j+3] > 0) { // if inpainted pixel is not transparent, replace the original pixel with the inpainted pixel
                if (true) {
                    inpainted_pixels[j] = original_pixels[j];
                    inpainted_pixels[j + 1] = original_pixels[j + 1];
                    inpainted_pixels[j + 2] = original_pixels[j + 2];
                } else { // if it is near the boundary, blend the inpainted pixel with the original pixel for smooth transition
                    inpainted_pixels[j] = Math.round((1*inpainted_pixels[j] + 0.0*original_pixels[j]));
                    inpainted_pixels[j + 1] = Math.round((1*inpainted_pixels[j + 1] + 0.0*original_pixels[j + 1]));
                    inpainted_pixels[j + 2] = Math.round((1*inpainted_pixels[j + 2] + 0.0*original_pixels[j + 2]));
                }
            }
        }

        inpainted_ctx.putImageData(inpainted_data, 0, 0);
        return inpainted_layer;
    }

    private current_coord_is_near_boundary() {

    }
    
    private preprocess_image(input: HTMLCanvasElement, width: number, height: number): ort.Tensor {
        const HW = width * height;

        // resize input to target dimensions
        const resized = document.createElement("canvas");
        resized.width = width;
        resized.height = height;
        const ctx = resized.getContext("2d")!;
        ctx.drawImage(input, 0, 0, width, height);

        // extract RGBA pixel data
        const image_data = ctx.getImageData(0, 0, width, height);
        const pixels = image_data.data;

        // create float32 array in CHW layout.
        const img_flat = new Float32Array(3 * HW);
        for (let i = 0; i < HW; i++) {
            const base = i * 4;
            img_flat[0 * HW + i] = pixels[base];
            img_flat[1 * HW + i] = pixels[base + 1];
            img_flat[2 * HW + i] = pixels[base + 2];
        }

        return new ort.Tensor("float32", img_flat, [1, 3, width, height]);
    }

    // FIXME: perf is bad for this function. Improve this later.
    public preprocess_mask(input: HTMLCanvasElement, width: number, height: number): ort.Tensor {
        // resize input to target dimensions
        const resized = document.createElement("canvas");
        resized.width = width;
        resized.height = height;
        const ctx = resized.getContext("2d")!;
        ctx.drawImage(input, 0, 0, width, height);

        // extract RGBA pixel data
        const image_data = ctx.getImageData(0, 0, width, height);
        const pixels = image_data.data;

        // if alpha > 0, set to 255. else 0.
        const normalized = new Float32Array(width * height);
        for (let i = 0; i < width * height; i++) {
            normalized[i] = pixels[i * 4 + 3] > 0 ? 255 : 0;
        }

        return new ort.Tensor("float32", normalized, [1, 1, width, height]);
    }

    public invert_tensor(tensor: ort.Tensor): ort.Tensor {
        const n = tensor.data.length;
        const isFloat = tensor.type === "float32";
        let inverted;

        if (isFloat) {
            inverted = new Float32Array(n);
            for (let i = 0; i < n; i++) {
                inverted[i] = 1.0 - (tensor.data[i] as number);
                inverted[i] = tensor.data[i] === 255 ? 0 : 255;
            }
        } else {
            inverted = new Uint8Array(n);
            for (let i = 0; i < n; i++) {
                inverted[i] = tensor.data[i] === 255 ? 0 : 255;
            }
        }

        return new ort.Tensor(tensor.type, inverted, tensor.dims);
    }

    private create_crop_mask(mask_a: ort.Tensor, mask_b: ort.Tensor): ort.Tensor {
        const SZ = 512;

        // combine both binary masks (1 if either has content)
        const combined = new Float32Array(SZ * SZ);
        for (let i = 0; i < SZ * SZ; i++) {
            combined[i] = (mask_a.data[i] as number > 0 || mask_b.data[i] as number > 0) ? 1 : 0;
        }

        const mat = new cv.Mat(SZ, SZ, cv.CV_8UC1);
        mat.data.set(combined);

        // dilate to push the crop boundary outward
        const kernel = cv.Mat.ones(7, 7, cv.CV_8U);
        const dilated = new cv.Mat();
        cv.dilate(mat, dilated, kernel);
        kernel.delete();
        mat.delete();

        // blur for smooth alpha transition at outer edge
        const blurred = new cv.Mat();
        cv.GaussianBlur(dilated, blurred, new cv.Size(1, 1), 10);
        dilated.delete();

        const feathered = new Float32Array(SZ * SZ);
        for (let i = 0; i < SZ * SZ; i++) {
            feathered[i] = blurred.data[i];
        }
        blurred.delete();

        return new ort.Tensor("float32", feathered, [1, 1, SZ, SZ]);
    }
    
}