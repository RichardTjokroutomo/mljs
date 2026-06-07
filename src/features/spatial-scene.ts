import type { inpaint } from "@techstark/opencv-js";
import { DepthEstimation } from "../models/depth-estimation.ts";
import { Inpaint } from "../models/inpaint.ts";
import * as ort from "onnxruntime-web/all";
import {html_image_to_html_canvas, html_canvas_to_ort_tensor, html_image_to_ort_tensor} from "../utils/type_converter.ts";

export class SpatialScene {
    depth_estimator: DepthEstimation;
    inpainter: Inpaint;

    public constructor(){
        // initialize objects
        this.depth_estimator = new DepthEstimation();
        this.inpainter = new Inpaint();

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
            executionProviders: [{
                name: "webgpu",
                device: "gpu",
            }, "wasm"],
            graphOptimizationLevel: "disabled",
        }
        const inpainter_options: ort.InferenceSession.SessionOptions = {
            executionProviders: [{
                name: "webgpu",
                device: "gpu",
            }, "wasm"],
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
        const processed_depth_estimation_result: HTMLCanvasElement = this.depth_estimator.postprocess(depth_estimation_result, target_img.width, target_img.height);

        // 4. segment image into layers
        const depth_map_ort_tensor: ort.Tensor = html_canvas_to_ort_tensor(processed_depth_estimation_result);
        const target_ort: ort.Tensor = html_image_to_ort_tensor(target_img);
        const layers: Array<ort.Tensor> = this.depth_estimator.segment_into_layers(target_ort, depth_map_ort_tensor, target_img.width, target_img.height, num_layers);

        // 5. inpaint each layer

        // 6. create canvas element for each inpainted layer
        
        // 7. add effects to each canvas elements

        // 8. remove original image element

        // 9. put the canvas elements. give the styling of the image elementss
    }
}