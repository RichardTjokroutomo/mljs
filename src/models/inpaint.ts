import type { BaseModel } from "./base-model.ts";
import * as cv from "@techstark/opencv-js";
import * as ort from "onnxruntime-web/all";
import { ort_tensor_to_html_canvas } from "../utils/type_converter.ts";

export class Inpaint implements BaseModel {
    ort_session: ort.InferenceSession | null = null;

    public constructor() {};

    public async create_session(model_path: Uint8Array, options: ort.InferenceSession.SessionOptions): Promise<void> {
        this.ort_session = await ort.InferenceSession.create(model_path, options);
    }

    public preprocess(input: Array<HTMLCanvasElement>, width: number, height: number): Array<ort.Tensor> {
        if (input.length != 2){
            throw new Error("input array is not 2, can't preprocess image for inpainting!");
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
    
    private preprocess_image(input: HTMLCanvasElement, width: number, height: number): ort.Tensor {
        const HW = width * height;
        const img_mat = cv.imread(input);
        let img_rgb = new cv.Mat();
        cv.cvtColor(img_mat, img_rgb, cv.COLOR_RGBA2RGB);
        let img_resized = new cv.Mat();
        cv.resize(img_rgb, img_resized, new cv.Size(width, height));
        img_mat.delete();
        img_rgb.delete();

        let img_flat = new Float32Array(3 * HW);
        for (let i = 0; i < HW; i++) {
            const base = i * 3;
            const ch_base = i; // spatial index per channel
            img_flat[0 * HW + ch_base] = img_resized.data[base]     / 255;
            img_flat[1 * HW + ch_base] = img_resized.data[base + 1] / 255;
            img_flat[2 * HW + ch_base] = img_resized.data[base + 2] / 255;
        }
        img_resized.delete();

        return new ort.Tensor("float32", img_flat, [1, 3, width, height]);
    }

    // FIXME: perf is bad for this function. Improve this later.
    private preprocess_mask(input: HTMLCanvasElement, width: number, height: number): ort.Tensor {
        const W = input.width;
        const H = input.height;

        const mat = cv.imread(input);

        // 1. if alpha is 0, set RGB to 0; otherwise set RGB to 255
        const total = W * H;
        for (let i = 0; i < total; i++) {
            const base = i * 4;
            if (mat.data[base + 3] === 0) {
                mat.data[base] = 0;
                mat.data[base + 1] = 0;
                mat.data[base + 2] = 0;
            } else {
                mat.data[base] = 255;
                mat.data[base + 1] = 255;
                mat.data[base + 2] = 255;
            }
        }

        // 2. convert to grayscale (all channels are identical after the step above)
        let gray = new cv.Mat();
        cv.cvtColor(mat, gray, cv.COLOR_RGBA2GRAY);
        mat.delete();

        // 3. resize
        let resized = new cv.Mat();
        cv.resize(gray, resized, new cv.Size(width, height));
        gray.delete();

        // 4. normalize to [0, 1] float32 range
        const normalized = new Float32Array(width * height);
        for (let i = 0; i < width * height; i++) {
            normalized[i] = resized.data[i] / 255;
        }
        resized.delete();

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
        const combined = new Uint8Array(SZ * SZ);
        for (let i = 0; i < SZ * SZ; i++) {
            combined[i] = (mask_a.data[i] as number > 0 || mask_b.data[i] as number > 0) ? 255 : 0;
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
            feathered[i] = blurred.data[i] / 255;
        }
        blurred.delete();

        return new ort.Tensor("float32", feathered, [1, 1, SZ, SZ]);
    }
    
}