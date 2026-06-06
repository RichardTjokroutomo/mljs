import type { BaseModel } from "./base-model.ts";
import * as cv from "@techstark/opencv-js";
import * as ort from "onnxruntime-web/all";

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
        const inverted_mask = this.invert_tensor(processed_mask);

        // return!
        return [processed_input_image, inverted_mask];
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

    public postprocess(input: ort.Tensor, width: number, height: number): HTMLCanvasElement {
        // TODO
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

    private invert_tensor(tensor: ort.Tensor): ort.Tensor {
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
}