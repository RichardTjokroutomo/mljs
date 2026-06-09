import cvModule from "@techstark/opencv-js";
import { DepthEstimation } from "../models/depth-estimation.ts";
import { Inpaint } from "../models/inpaint.ts";
import { Animation } from "../ui/animation.ts";
import * as ort from "onnxruntime-web/all";
import {html_image_to_html_canvas, html_canvas_to_html_image, ort_tensor_to_html_canvas} from "../utils/type_converter.ts";
import { resize_html_canvas } from "../utils/html_canvas_manipulator.ts";

const cv = (cvModule as any).default ?? cvModule; // Handle both default and named exports from OpenCV.js

export class SpatialScene {
    depth_estimator: DepthEstimation;
    inpainter: Inpaint;
    animation: Animation;

    public constructor(){
        // initialize objects
        this.depth_estimator = new DepthEstimation();
        this.inpainter = new Inpaint();
        this.animation = new Animation();

        // TODO: load models & packages

    }

    public async initialize_sessions(depth_estimation_path: string, inpaint_path: string){
        // specify model paths. (TODO: fix this once package_loader has been implemented)
        const depth_estimation_res = await fetch(depth_estimation_path);
        const depth_estimation_array_buffer = await depth_estimation_res.arrayBuffer();
        const depth_estimation_uint8_array = new Uint8Array(depth_estimation_array_buffer);

        const inpainter_res = await fetch(inpaint_path);
        const inpainter_array_buffer = await inpainter_res.arrayBuffer();
        const inpainter_uint8_array = new Uint8Array(inpainter_array_buffer);

        // session options (TODO: client JS must be able to specify this)
        const depth_estimator_options: ort.InferenceSession.SessionOptions = {
            executionProviders: ["webgpu", "wasm"],
            graphOptimizationLevel: "disabled",
        }
        const inpainter_options: ort.InferenceSession.SessionOptions = {
            executionProviders: ["webgpu", "wasm"],
        };

        // create sessions
        await this.depth_estimator.create_session(depth_estimation_uint8_array, depth_estimator_options);
        await this.inpainter.create_session(inpainter_uint8_array, inpainter_options);
    }

    // TODO: for now, client JS will call this code. but in the future, make this private and declare a wrapper function.
    public async convert_single_image(input_container: HTMLDivElement, num_layers: number){
        // 0. local vars
        const DEPTH_ESTIMATION_INPUT_WIDTH = 518; // DA_v2's input width
        const DEPTH_ESTIMATION_INPUT_HEIGHT = 518;
        const INPAINT_INPUT_WIDTH = 512; // MI-GAN's input width
        const INPAINT_INPUT_HEIGHT = 512;

        // 1. rule check & obtain image element
        if ((input_container.childElementCount != 1) || 
        (input_container.children[0].tagName != "IMG"))
        {
            console.log("cannot apply spatial scene!");
            return;
        }

        const target_img: HTMLImageElement = input_container.children[0] as HTMLImageElement;

        // 2. convert image to canvas
        let target_canvas: HTMLCanvasElement = html_image_to_html_canvas(target_img);
        target_canvas = resize_html_canvas(target_canvas, 518, 518);

        // 3. perform depth estimation
        const depth_estimation_input: Array<ort.Tensor> = this.depth_estimator.preprocess([target_canvas], DEPTH_ESTIMATION_INPUT_WIDTH, DEPTH_ESTIMATION_INPUT_HEIGHT);
        const depth_estimation_result: ort.Tensor = await this.depth_estimator.run_inference(depth_estimation_input);
        const processed_depth_estimation_result: HTMLCanvasElement = this.depth_estimator.postprocess([depth_estimation_result], DEPTH_ESTIMATION_INPUT_WIDTH, DEPTH_ESTIMATION_INPUT_HEIGHT); // TODO: dim should be the original dim

        // 4. segment image into layers
        const layers: Array<HTMLCanvasElement> = this.depth_estimator.segment_into_layers(target_canvas, processed_depth_estimation_result, DEPTH_ESTIMATION_INPUT_WIDTH, DEPTH_ESTIMATION_INPUT_HEIGHT, num_layers); // TODO: dim should be the original dim

        // 5. inpaint each layer
        let inpainted_layers: Array<HTMLCanvasElement> = [];
        target_canvas = resize_html_canvas(target_canvas, INPAINT_INPUT_WIDTH, INPAINT_INPUT_HEIGHT);
        for (let i: number = num_layers-1; i >= 0; i--){
            if (i == num_layers-1){
                inpainted_layers.push(layers[i]);
            } else {
                const last_inpainted_layer: HTMLCanvasElement = inpainted_layers[inpainted_layers.length - 1];
                
                target_canvas = resize_html_canvas(target_canvas, INPAINT_INPUT_WIDTH, INPAINT_INPUT_HEIGHT);
                const inpainter_inputs: Array<ort.Tensor> = this.inpainter.preprocess([target_canvas, last_inpainted_layer, layers[i]], INPAINT_INPUT_WIDTH, INPAINT_INPUT_HEIGHT);
                const inverted_mask: ort.Tensor = this.inpainter.invert_tensor(inpainter_inputs[1]);

                const inpainted_result: ort.Tensor = await this.inpainter.run_inference([inpainter_inputs[0], inverted_mask]);
                
                const processed_inpainted_result: HTMLCanvasElement = this.inpainter.postprocess([inpainted_result, inpainter_inputs[1], inpainter_inputs[2]], INPAINT_INPUT_WIDTH, INPAINT_INPUT_HEIGHT);
                inpainted_layers.push(processed_inpainted_result);

            }
        }


        // 6. convert each canvas back to image & add effects
        let inpainted_images: Array<HTMLImageElement> = [];
        const parallax_factors = [0.055, 0.065, 0.075, 0.09]; // TODO: make this user argument.
        for (let i: number = 0; i < inpainted_layers.length; i++){
            let image_layer = html_canvas_to_html_image(inpainted_layers[i]);

            inpainted_images.push(this.animation.add_cursor_hover_effect(input_container, image_layer, parallax_factors[i]));
        }

        // 8. Save container dimensions and ensure position:relative before removing original image
        const containerWidth = input_container.clientWidth;
        const containerHeight = input_container.clientHeight;
        input_container.style.position = "relative";
        input_container.removeChild(input_container.firstChild as HTMLImageElement);

        // 9. put the canvas elements & center their midpoint with the container's midpoint
        for (let i: number = 0; i < inpainted_images.length; i++){
            inpainted_images[i].style.zIndex = `${-i}`;
            inpainted_images[i].style.position = "absolute";
            // Use the source canvas dimensions — image.naturalWidth/Height is 0
            // since the Image hasn't loaded asynchronously yet
            const canvasWidth = inpainted_layers[i].width;
            const canvasHeight = inpainted_layers[i].height;
            inpainted_images[i].style.left = `${(containerWidth - canvasWidth) / 2}px`;
            inpainted_images[i].style.top = `${(containerHeight - canvasHeight) / 2}px`;
            inpainted_images[i].style.transform = "scale(1.1)";
            inpainted_images[i].style.width = `${canvasWidth}px`;
            inpainted_images[i].style.height = `${canvasHeight}px`;
            input_container.appendChild(inpainted_images[i]);
        }
    }
}