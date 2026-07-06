/*
    Inpainter implementation using WebNN API. 
    Currently, the typing is not specified well because base typescript doesn't support webNN types yet.


    Conventions: 
        - For masks, 0 means keep; 1 means inpaint / discard.
*/

import { WeightsFile, buildGraph } from "../../../model_binaries/webnn/migan/migan.js";

export class Inpaint {
    graph: any | null = null;
    context: any | null = null;
    weights: any | null = null;
    width: number;
    height: number;

    public constructor(){
        if (!navigator.ml) throw new Error("WebNN not supported!");
        this.width = 512;
        this.height = 512;
    }

    public async create_session(device_type: string){
        const base_dir: string = "../../model_binaries/webnn/migan/";
        this.context = await navigator.ml.createContext({ device_type });
        this.weights = await WeightsFile.load(base_dir + "migan.weights", base_dir + "migan.manifest.json");
        this.graph = await buildGraph(this.context, this.weights);
    }

    public async run_inference(inputs: Float32Array): Promise<Float32Array> {
        const input_webnn_tensor = await this.context.createTensor({dataType: "float32", shape: [1, 4, this.width, this.height], writable: true});
        const output_webnn_tensor = await this.context.createTensor({dataType: "float32", shape: [1, 3, this.width, this.height], readable: true});

        this.context.writeTensor(input_webnn_tensor, inputs);
        const input_tensors = {
            "serving_default_args_0": input_webnn_tensor,
        };
        const output_tensors = {
            "serving_default_output_0_output": output_webnn_tensor,
        };

        this.context.dispatch(this.graph.graph, input_tensors, output_tensors);

        const result: Float32Array = new Float32Array(await this.context.readTensor(output_webnn_tensor));

        return result;
    }

    public preprocess(inputs: Array<HTMLCanvasElement>, width: number = this.width, height: number = this.height): Array<Float32Array> {
        const input_array: Float32Array = this.preprocess_input(inputs[0], width, height);
        const mask_array: Float32Array = this.preprocess_mask(inputs[1], width, height);
        const current_layer_array: Float32Array = this.preprocess_mask(inputs[2], width, height);

        const combined_array: Float32Array = this.merge_inputs(input_array, mask_array, width, height);
        const combined_mask: Float32Array = this.merge_masks(current_layer_array, mask_array);

        return [combined_array, combined_mask, mask_array];
    }

    public postprocess(inputs: Array<Float32Array>, width: number = this.width, height: number = this.height): HTMLCanvasElement {
        const wh: number = width * height;

        // combine inpainted area & original
        for (let i: number = 0; i < wh; i++){
            inputs[0][0*wh + i] = inputs[2][0*wh + i]*inputs[3][i] + inputs[0][0*wh + i]*(1 - inputs[3][i]);
            inputs[0][1*wh + i] = inputs[2][1*wh + i]*inputs[3][i] + inputs[0][1*wh + i]*(1 - inputs[3][i]);
            inputs[0][2*wh + i] = inputs[2][2*wh + i]*inputs[3][i] + inputs[0][2*wh + i]*(1 - inputs[3][i]);
        }

        // normalize inpainted layer & apply mask
        let normalized_layer: Uint8ClampedArray = new Uint8ClampedArray(4 * wh);

        for (let i: number = 0; i < wh; i++){
            const base: number = 4 * i;

            normalized_layer[base + 0] = (inputs[0][0*wh + i] + 1) * 127.5;
            normalized_layer[base + 1] = (inputs[0][1*wh + i] + 1) * 127.5;
            normalized_layer[base + 2] = (inputs[0][2*wh + i] + 1) * 127.5;

            if (inputs[1][i] === 0){
                normalized_layer[base + 3] = 255;
            } else {
                normalized_layer[base + 3] = 0;
            }
        }

        // convert to canvas
        const image_data: ImageData = new ImageData(normalized_layer, width, height);
        const canvas: HTMLCanvasElement = document.createElement("canvas");
        canvas.width = width;
        canvas.height = height;

        const ctx: CanvasRenderingContext2D = canvas.getContext("2d")!;
        ctx.putImageData(image_data, 0, 0);

        // return!
        return canvas;
    }

    public preprocess_input(input: HTMLCanvasElement, width: number, height: number): Float32Array {
        const wh: number = width * height;

        // resize
        const resized_canvas: HTMLCanvasElement = document.createElement("canvas");
        resized_canvas.width = width;
        resized_canvas.height = height;
        const ctx: CanvasRenderingContext2D = resized_canvas.getContext("2d")!;
        ctx.drawImage(input, 0, 0, width, height);

        // get pixel data
        const image_data: ImageData = ctx.getImageData(0, 0, width, height);
        const pixels: ImageDataArray = image_data.data;

        // create fp32 array in channel first format
        let input_array: Float32Array = new Float32Array(3 * wh);
        for (let i: number = 0; i < wh; i++){
            const base: number = i * 4;
            input_array[0*wh + i] = (pixels[base] / 127.5) - 1;
            input_array[1*wh + i] = (pixels[base + 1] / 127.5) - 1;
            input_array[2*wh + i] = (pixels[base + 2] / 127.5) - 1;
        }

        return input_array;
    }

    private preprocess_mask(mask: HTMLCanvasElement, width: number, height: number): Float32Array {
        const wh: number = width * height;

        // resize
        const resized_canvas: HTMLCanvasElement = document.createElement("canvas");
        resized_canvas.width = width;
        resized_canvas.height = height;
        const ctx: CanvasRenderingContext2D = resized_canvas.getContext("2d")!;
        ctx.drawImage(mask, 0, 0, width, height);

        // get pixel data
        const mask_data: ImageData = ctx.getImageData(0, 0, width, height);
        const pixels: ImageDataArray = mask_data.data;

        // create fp32 array. If alpha == 0, then 1. (transparent means keep)
        let mask_array: Float32Array = new Float32Array(wh);
        for (let i: number = 0; i < wh; i++){
            if (pixels[i*4 + 3] === 0){
                mask_array[i] = 1;
            } else {
                mask_array[i] = 0;
            }
        }

        return mask_array;
    }

    private merge_inputs(input_array: Float32Array, mask_array: Float32Array, width: number, height: number): Float32Array {
        // formula: concat(mask-0.5, input * mask)
        // range: mask [0, 1]; input [-1, 1]
        const wh: number = width * height;
        let combined_array: Float32Array = new Float32Array(4 * wh);

        // insert mask
        for (let i: number = 0; i < wh; i++){
            combined_array[i] = mask_array[i] - 0.5;
        }

        // insert input
        for (let channel: number = 0; channel < 3; channel++){
            for (let i: number = 0; i < wh; i++){
                combined_array[(channel + 1)*wh + i] = input_array[channel*wh + i] * mask_array[i];
            }
        }

        return combined_array;
    }

    // TODO: try to merge this function with merge_inputs() in the future
    private merge_masks(mask_a: Float32Array, mask_b: Float32Array): Float32Array {
        if (mask_a.length != mask_b.length) throw new Error("Cannot merge mask layers as their dimension(s) is(are) different!");

        let combined_mask: Float32Array = new Float32Array(mask_a.length);

        for (let i: number = 0; i < mask_a.length; i++){
            if (mask_a[i] === 0 || mask_b[i] === 0){
                combined_mask[i] = 0;
            } else {
                combined_mask[i] = 1;
            }
        }

        return combined_mask;
    }
}