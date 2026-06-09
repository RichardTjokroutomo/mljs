import cvModule from "@techstark/opencv-js";
import { DepthEstimation } from "../models/depth-estimation.ts";
import { Inpaint } from "../models/inpaint.ts";
import { Animation } from "../ui/animation.ts";
import * as ort from "onnxruntime-web/all";
import {html_image_to_html_canvas, html_canvas_to_html_image} from "../utils/type_converter.ts";

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
        console.log("Initializing sessions with model paths:", depth_estimation_path, inpaint_path);
        const res = await fetch(depth_estimation_path);
        console.log("Fetched depth estimation model, response:", res);
        
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
        const TOTAL_TIME_0 = Date.now();
        // 1. rule check & obtain image element
        if ((input_container.childElementCount != 1) || 
        (input_container.children[0].tagName != "IMG"))
        {
            console.log("cannot apply spatial scene!");
            return;
        }

        const target_img: HTMLImageElement = input_container.children[0] as HTMLImageElement;
        console.log("step 1 done!");

        // 2. convert image to canvas
        const target_canvas: HTMLCanvasElement = html_image_to_html_canvas(target_img);
        console.log("step 2 done!");

        // 3. perform depth estimation
        const DA_1 = Date.now();
        const depth_estimation_input: Array<ort.Tensor> = this.depth_estimator.preprocess([target_canvas], 518, 518);
        const DA_2 = Date.now();
        const depth_estimation_result: ort.Tensor = await this.depth_estimator.run_inference(depth_estimation_input);
        const DA_3 = Date.now();
        const processed_depth_estimation_result: HTMLCanvasElement = this.depth_estimator.postprocess([depth_estimation_result], 518, 518); // TODO: dim should be the original dim
        const DA_4 = Date.now();
        // TODO: remove this once debugging is complete.
        // const depth_link = document.createElement("a");
        // depth_link.download = "depth_map.png";
        // depth_link.href = processed_depth_estimation_result.toDataURL("image/png");
        // depth_link.click();

        console.log(`Depth estimation timings (ms): preprocess ${DA_2 - DA_1}, inference ${DA_3 - DA_2}, postprocess ${DA_4 - DA_3}`);
        console.log("step 3 done!");
        
        // 4. segment image into layers
        const DA_5 = Date.now();
        const layers: Array<HTMLCanvasElement> = this.depth_estimator.segment_into_layers(target_canvas, processed_depth_estimation_result, 518, 518, num_layers); // TODO: dim should be the original dim
        const DA_6 = Date.now();
        console.log(`Segmentation timings (ms): ${DA_6 - DA_5}`);
        console.log("step 4 done!");

        // TODO: remove this once debugging is complete.
        // for (let i = 0; i < layers.length; i++) {
        //     const lnk = document.createElement("a");
        //     lnk.download = `t_layer_${i}.png`;
        //     lnk.href = layers[i].toDataURL("image/png");
        //     lnk.click();
        // }

        // 5. inpaint each layer
        let inpainted_layers: Array<HTMLCanvasElement> = [];
        for (let i: number = num_layers-1; i >= 0; i--){
            console.log("================================");
            console.log(`Inpainting layer ${num_layers-1 - i}...`);
            if (i == num_layers-1){
                inpainted_layers.push(layers[i]);
            } else {
                const IP_1 = Date.now();
                const last_inpainted_layer: HTMLCanvasElement = inpainted_layers[inpainted_layers.length - 1];
                
                const inpainter_inputs: Array<ort.Tensor> = this.inpainter.preprocess([target_canvas, last_inpainted_layer, layers[i]], 512, 512);
                const inverted_mask: ort.Tensor = this.inpainter.invert_tensor(inpainter_inputs[1]);
                const IP_2 = Date.now();
                

                // TODO: remove this once debugging is complete.
                // const inpaint_input_canvas = ort_tensor_to_html_canvas(inpainter_inputs[0]);
                // const inpaint_input_lnk = document.createElement("a");
                // inpaint_input_lnk.download = `u_inpaint_input_${i}.png`;
                // inpaint_input_lnk.href = inpaint_input_canvas.toDataURL("image/png");
                // inpaint_input_lnk.click();

                // const mask_data = inverted_mask.data as Float32Array;
                // const mask_canvas = document.createElement("canvas");
                // mask_canvas.width = 512;
                // mask_canvas.height = 512;
                // const mask_ctx = mask_canvas.getContext("2d")!;
                // const mask_img = mask_ctx.createImageData(512, 512);
                // const mask_pixels = mask_img.data;
                // for (let p = 0; p < 512 * 512; p++) {
                //     const v = Math.round(mask_data[p] * 255);
                //     const j = p * 4;
                //     mask_pixels[j] = v;
                //     mask_pixels[j + 1] = v;
                //     mask_pixels[j + 2] = v;
                //     mask_pixels[j + 3] = 255;
                // }
                // mask_ctx.putImageData(mask_img, 0, 0);
                // const mask_lnk = document.createElement("a");
                // mask_lnk.download = `u_inverted_mask_${i}.png`;
                // mask_lnk.href = mask_canvas.toDataURL("image/png");
                // mask_lnk.click();
                


                const inpainted_result: ort.Tensor = await this.inpainter.run_inference([inpainter_inputs[0], inverted_mask]);
                const IP_3 = Date.now();


                // TODO: remove this once debugging is complete.
                // const i_data = inpainted_result.data as Uint8Array;
                // const i_dims = inpainted_result.dims;
                // const i_h = i_dims[2];
                // const i_w = i_dims[3];
                // const i_canvas = document.createElement("canvas");
                // i_canvas.width = i_w;
                // i_canvas.height = i_h;
                // const i_ctx = i_canvas.getContext("2d")!;
                // const i_img_data = i_ctx.createImageData(i_w, i_h);
                // const i_pixels = i_img_data.data;
                // for (let iy = 0; iy < i_h; iy++) {
                //     for (let ix = 0; ix < i_w; ix++) {
                //         const pi = (iy * i_w + ix) * 4;
                //         i_pixels[pi]     = i_data[0 * i_h * i_w + iy * i_w + ix];
                //         i_pixels[pi + 1] = i_data[1 * i_h * i_w + iy * i_w + ix];
                //         i_pixels[pi + 2] = i_data[2 * i_h * i_w + iy * i_w + ix];
                //         i_pixels[pi + 3] = 255;
                //     }
                // }
                // i_ctx.putImageData(i_img_data, 0, 0);
                // const inpainted_lnk = document.createElement("a");
                // inpainted_lnk.download = `inpainted_layer_${i}.png`;
                // inpainted_lnk.href = i_canvas.toDataURL("image/png");
                // inpainted_lnk.click();

                
                const processed_inpainted_result: HTMLCanvasElement = this.inpainter.postprocess([inpainted_result, inpainter_inputs[1], inpainter_inputs[2]], 512, 512);
                const IP_4 = Date.now();
                inpainted_layers.push(processed_inpainted_result);

                console.log(`Inpainting timings (ms): preprocess ${IP_2 - IP_1}, inference ${IP_3 - IP_2}, postprocess ${IP_4 - IP_3}`);
            }
        }
        console.log("step 5 done!");

        // TODO: remove this once debugging is complete.
        // for (let i = 0; i < inpainted_layers.length; i++) {
        //     const lnk = document.createElement("a");
        //     lnk.download = `u_inpainted_${i}.png`;
        //     lnk.href = inpainted_layers[i].toDataURL("image/png");
        //     lnk.click();
        // }

        // 6. convert each canvas back to image & add effects
        let inpainted_images: Array<HTMLImageElement> = [];
        const parallax_factors = [0.055, 0.065, 0.075, 0.08]; // TODO: make this user argument.
        for (let i: number = 0; i < inpainted_layers.length; i++){
            let image_layer = html_canvas_to_html_image(inpainted_layers[i]);

            inpainted_images.push(this.animation.add_cursor_hover_effect(input_container, image_layer, parallax_factors[i]));
        }
        console.log("step 6 done!");
        

        // 8. Save container dimensions and ensure position:relative before removing original image
        const containerWidth = input_container.clientWidth;
        const containerHeight = input_container.clientHeight;
        input_container.style.position = "relative";
        input_container.removeChild(input_container.firstChild as HTMLImageElement);
        console.log("step 7 done!");

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
        const TOTAL_TIME_1 = Date.now();
        console.log(`Total processing time (ms): ${TOTAL_TIME_1 - TOTAL_TIME_0}`);
        console.log("step 8 done!");
    }
}