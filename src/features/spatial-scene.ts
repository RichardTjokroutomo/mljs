import * as cv from "@techstark/opencv-js";
import { DepthEstimation } from "../models/depth-estimation.ts";
import { Inpaint } from "../models/inpaint.ts";
import { Animation } from "../ui/animation.ts";
import * as ort from "onnxruntime-web/all";
import {html_image_to_html_canvas, html_canvas_to_ort_tensor, html_image_to_ort_tensor, ort_tensor_to_html_canvas, ort_tensor_to_cv_mat, html_image_to_cv_mat, html_canvas_to_html_image} from "../utils/type_converter.ts";

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

    public async initialize_sessions(){
        // specify model paths. (TODO: fix this once package_loader has been implemented)
        const depth_estimator_path: string = "../../models/depth_anything_v2_vits_quantized.onnx";
        const inpainter_path: string = "../../models/migan_processed.onnx";

        const encoder = new TextEncoder();
        const depth_estimator_uint8_path: Uint8Array = encoder.encode(depth_estimator_path);
        const inpainter_uint8_path: Uint8Array = encoder.encode(inpainter_path);

        // session options (TODO: client JS must be able to specify this)
        const depth_estimator_options: ort.InferenceSession.SessionOptions = {
            executionProviders: ["webgpu", "wasm"],
            graphOptimizationLevel: "disabled",
        }
        const inpainter_options: ort.InferenceSession.SessionOptions = {
            executionProviders: ["webgpu", "wasm"],
        };

        // create sessions
        await this.depth_estimator.create_session(depth_estimator_uint8_path, depth_estimator_options);
        await this.inpainter.create_session(inpainter_uint8_path, inpainter_options);
    }

    // TODO: for now, client JS will call this code. but in the future, make this private and declare a wrapper function.
    public async convert_single_image(input_container: HTMLDivElement, class_name: string, num_layers: number){
        // 1. rule check & obtain image element
        if ((input_container.childElementCount != 1) || 
        (input_container.children[0].tagName != "IMG") || 
        (input_container.children[0].className != class_name))
        {
            console.log("cannot apply spatial scene!");
            return;
        }

        const target_img: HTMLImageElement = input_container.children[0] as HTMLImageElement;

        // 2. convert image to canvas
        const target_canvas: HTMLCanvasElement = html_image_to_html_canvas(target_img);

        // 3. perform depth estimation
        const depth_estimation_input: Array<ort.Tensor> = this.depth_estimator.preprocess([target_canvas], 518, 518);
        const depth_estimation_result: ort.Tensor = await this.depth_estimator.run_inference(depth_estimation_input);
        const processed_depth_estimation_result: HTMLCanvasElement = this.depth_estimator.postprocess([depth_estimation_result], target_img.width, target_img.height);

        // 4. segment image into layers
        const layers: Array<HTMLCanvasElement> = this.depth_estimator.segment_into_layers(target_canvas, processed_depth_estimation_result, target_img.width, target_img.height, num_layers);

        // 5. inpaint each layer
        let inpainted_layers: Array<HTMLCanvasElement> = [];
        for (let i: number = num_layers-1; i >= 0; i--){
            if (i == num_layers-1){
                inpainted_layers.push(layers[i]);
            } else {
                const last_inpainted_layer: HTMLCanvasElement = inpainted_layers[inpainted_layers.length - 1];
                
                const inpainter_inputs: Array<ort.Tensor> = this.inpainter.preprocess([target_canvas, last_inpainted_layer, layers[i]], 512, 512);
                const inverted_mask: ort.Tensor = this.inpainter.invert_tensor(inpainter_inputs[1]);
                const inpainted_result: ort.Tensor = await this.inpainter.run_inference([inpainter_inputs[0], inverted_mask]);
                const processed_inpainted_result: HTMLCanvasElement = this.inpainter.postprocess([inpainted_result, inpainter_inputs[1], inpainter_inputs[2]], 512, 512);
                inpainted_layers.push(processed_inpainted_result);
            }
        }

        // 6. convert each canvas back to image & add effects
        let inpainted_images: Array<HTMLImageElement> = [];
        const parallax_factors = [0.055, 0.065, 0.075, 0.08]; // TODO: make this user argument.
        for (let i: number = 0; i < inpainted_layers.length; i++){
            let image_layer = html_canvas_to_html_image(inpainted_layers[i]);

            inpainted_images.push(this.animation.add_cursor_hover_effect(input_container, image_layer, parallax_factors[i]));
        }
        

        // 8. remove original image element
        input_container.removeChild(input_container.firstChild as HTMLImageElement);

        // 9. put the canvas elements. give the styling of the image elements
        for (let i: number = 0; i < inpainted_images.length; i++){
            inpainted_images[i].style.zIndex = `${-i}`;
            inpainted_images[i].style.position = "absolute";
            const imgWidth = inpainted_images[i].width;
            const imgHeight = inpainted_images[i].height;
            inpainted_images[i].style.left = `${(input_container.clientWidth - imgWidth) / 2}px`;
            inpainted_images[i].style.top = `${(input_container.clientHeight - imgHeight) / 2}px`;
            inpainted_images[i].style.transform = "scale(1.1)";
            input_container.appendChild(inpainted_images[i]);
        }
    }
}