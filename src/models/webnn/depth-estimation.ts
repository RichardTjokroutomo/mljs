import { WeightsFile, buildGraph } from "../../../model_binaries/webnn/depth-anything-v2/depth_anything_v2_quantized.js";

export class DepthEstimation {
    graph: any | null = null;
    context: any | null = null;
    weights: any | null = null;
    width: number;
    height: number;

    public constructor(){
        if (!navigator.ml) throw new Error("WebNN not supported!");
        this.width = 518;
        this.height = 518;
    }

    public async create_session(device_type: string){
        const base_dir: string = "../../model_binaries/webnn/depth-anything-v2/";
        this.context = await navigator.ml.createContext({ device_type });
        this.weights = await WeightsFile.load(base_dir + "depth_anything_v2_quantized.weights", base_dir + "depth_anything_v2_quantized.manifest.json");
        this.graph = await buildGraph(this.context, this.weights);
    }

    public async run_inference(input: Float32Array): Promise<Float32Array> {
        const input_webnn_tensor = await this.context.createTensor({dataType: "float32", shape: [1, 3, this.width, this.height], writable: true});
        const output_webnn_tensor = await this.context.createTensor({dataType: "float32", shape: [1, this.width, this.height], readable: true});

        this.context.writeTensor(input_webnn_tensor, input);

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

    public preprocess(input: HTMLCanvasElement, width: number = this.width, height: number = this.height): Float32Array {
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

        // create fp32 channel-first array
        let input_array: Float32Array = new Float32Array(3 * wh);
        for(let i: number = 0; i < wh; i++){
            const base: number = i * 4;
            input_array[0*wh + i] = (pixels[base] / 127.5) - 1;
            input_array[1*wh + i] = (pixels[base + 1] / 127.5) - 1;
            input_array[2*wh + i] = (pixels[base + 2] / 127.5) - 1;
        }

        return input_array;
    }

    public postprocess(input: Float32Array, width: number = this.width, height: number = this.height): HTMLCanvasElement {
        const wh: number = width * height;
        const scale_factor: number = 255;

        // find largest value
        let largest_pixel_value: number = input[0];
        for (let i: number = 0; i < wh; i++){
            if(input[i] > largest_pixel_value){
                largest_pixel_value = input[i];
            }
        }

        // create canvas
        let normalized_input: Uint8ClampedArray = new Uint8ClampedArray(4 * wh);
        for (let i: number = 0; i < wh; i++){
            const base: number = i* 4;
            normalized_input[base] = (input[i] / largest_pixel_value) * scale_factor;
            normalized_input[base + 1] = (input[i] / largest_pixel_value) * scale_factor;
            normalized_input[base + 2] = (input[i] / largest_pixel_value) * scale_factor;
            normalized_input[base + 3] = 255;
        }

        const image_data: ImageData = new ImageData(normalized_input, width, height);
        const canvas: HTMLCanvasElement = document.createElement("canvas");
        canvas.width = width;
        canvas.height = height;

        const ctx: CanvasRenderingContext2D = canvas.getContext("2d")!;
        ctx.putImageData(image_data, 0, 0);

        return canvas;
    }

    public segment_into_layers(
        input: HTMLCanvasElement, 
        depth_map: HTMLCanvasElement, 
        width: number = this.width, 
        height: number = this.height, 
        num_layers: number
    ): Array<HTMLCanvasElement>{
        const input_ctx = input.getContext("2d")!;
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