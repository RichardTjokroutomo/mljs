import * as ort from "onnxruntime-web/all";
import cvModule from "@techstark/opencv-js";
import { ort_tensor_to_html_canvas } from "./type_converter.ts";

const cv = (cvModule as any).default ?? cvModule;

function trigger_download(canvas: HTMLCanvasElement, filename: string): void {
    const link = document.createElement("a");
    link.download = filename;
    link.href = canvas.toDataURL("image/png");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

export function download_tensor(tensor: ort.Tensor, filename: string): void {
    const canvas = ort_tensor_to_html_canvas(tensor);
    trigger_download(canvas, filename);
}

export function download_canvas(canvas: HTMLCanvasElement, filename: string): void {
    trigger_download(canvas, filename);
}

export function download_cv_mat(mat: cv.Mat, filename: string): void {
    const canvas = document.createElement("canvas");
    cv.imshow(canvas, mat);
    trigger_download(canvas, filename);
}

export function array_to_html_canvas(input: Float32Array, width: number, height: number, is_three_channel: boolean){
    let wh = width * height;
    const scale_factor = 1;

    const clamped_data = new Uint8ClampedArray(4 * wh);
    for (let i = 0; i < wh; i++){
        const base = 4 * i;
        clamped_data[base + 0] = input[0*wh + i] * scale_factor;
        clamped_data[base + 1] = input[1*wh + i] * scale_factor;
        clamped_data[base + 2] = input[2*wh + i] * scale_factor;
        clamped_data[base + 3] = 255;

        if (!is_three_channel){
            clamped_data[base + 3] = input[3*wh + i];
        }
    }

    const image_data = new ImageData(clamped_data, width, height);

    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;

    const ctx = canvas.getContext("2d");
    ctx.putImageData(image_data, 0, 0);

    return canvas;
}

export function download_array(input: Float32Array, filename: string){
    const canvas = array_to_html_canvas(input, 512, 512, false);
    download_canvas(canvas, filename);
}