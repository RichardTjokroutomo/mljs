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
