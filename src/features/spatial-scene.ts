import type { inpaint } from "@techstark/opencv-js";
import { DepthEstimation } from "../models/depth-estimation.ts";
import { Inpaint } from "../models/inpaint.ts";
import * as ort from "onnxruntime-web/all";

export class SpatialScene {
    depth_estimator: DepthEstimation;
    inpainter: Inpaint;

    public constructor(){
        // initialize objects
        this.depth_estimator = new DepthEstimation();
        this.inpainter = new Inpaint();

        // TODO: load models & packages

        // specify model paths. (TODO: fix this once package_loader has been implemented)
        const depth_estimator_path: string = "./models/depth_anything_v2_vits_quantized.onnx";
        const inpainter_path: string = "./models/migan_processed.onnx";

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
        this.depth_estimator.create_session(depth_estimator_uint8_path, depth_estimator_options);
        this.inpainter.create_session(inpainter_uint8_path, inpainter_options);

    }

    // TODO: for now, client JS will call this code. but in the future, make this private and declare a wrapper function.
    public convert_single_image(input_div: HTMLDivElement){
        // 1. obtain image element

        // 2. convert image to canvas

        // 3. perform depth estimation

        // 3.2 segment image into layers

        // 4. inpaint each layer

        // 5. create canvas element for each inpainted layer
        
        // 6. add effects to each canvas elements

        // 7. remove original image element

        // 8. put the canvas elements. give the styling of the image elementss
    }
}