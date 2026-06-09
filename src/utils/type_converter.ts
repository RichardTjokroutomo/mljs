import * as ort from "onnxruntime-web/all";
import * as cv from "@techstark/opencv-js";

// from canvas
// ============================================================================================
export function html_canvas_to_html_image(canvas: HTMLCanvasElement): HTMLImageElement {
    const image = new Image();
    image.src = canvas.toDataURL();
    return image;
}

export function html_canvas_to_ort_tensor(canvas: HTMLCanvasElement): ort.Tensor {
    const ctx = canvas.getContext("2d");
    if (!ctx) {
        throw new Error("failed to get 2d context from canvas");
    }
    const image_data = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const { data, width, height } = image_data;
    const channels = 3;
    const float_data = new Float32Array(channels * height * width);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const pixel_idx = (y * width + x) * 4;
            for (let c = 0; c < channels; c++) {
                float_data[c * height * width + y * width + x] = data[pixel_idx + c]! / 255;
            }
        }
    }
    return new ort.Tensor("float32", float_data, [1, channels, height, width]);
}

export function html_canvas_to_cv_mat(canvas: HTMLCanvasElement): cv.Mat {
    return cv.imread(canvas);
}

// from image
// ============================================================================================
export function html_image_to_html_canvas(image: HTMLImageElement): HTMLCanvasElement {
    const canvas = document.createElement("canvas");
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
        throw new Error("failed to get 2d context");
    }
    ctx.drawImage(image, 0, 0);
    return canvas;
}

export function html_image_to_ort_tensor(image: HTMLImageElement): ort.Tensor {
    const canvas = html_image_to_html_canvas(image);
    return html_canvas_to_ort_tensor(canvas);
}

export function html_image_to_cv_mat(image: HTMLImageElement): cv.Mat {
    const canvas = html_image_to_html_canvas(image);
    return html_canvas_to_cv_mat(canvas);
}

// from ort tensor
// ============================================================================================
export function ort_tensor_to_html_canvas(tensor: ort.Tensor): HTMLCanvasElement {
    const dims = tensor.dims;
    if (dims.length !== 4 || dims[0] !== 1 || dims[1] !== 3) {
        throw new Error("expected tensor shape [1, 3, H, W]");
    }
    const [, , height, width] = dims;
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
        throw new Error("failed to get 2d context");
    }
    const image_data = ctx.createImageData(width, height);
    const is_float = tensor.type === "float32";
    if (is_float) {
        const tensor_data = tensor.data as Float32Array;
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const pixel_idx = (y * width + x) * 4;
                for (let c = 0; c < 3; c++) {
                    image_data.data[pixel_idx + c] = Math.max(
                        0,
                        Math.min(255, Math.round(tensor_data[c * height * width + y * width + x]! * 255)),
                    );
                }
                image_data.data[pixel_idx + 3] = 255;
            }
        }
    } else {
        const tensor_data = tensor.data as Uint8Array;
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const pixel_idx = (y * width + x) * 4;
                for (let c = 0; c < 3; c++) {
                    image_data.data[pixel_idx + c] = tensor_data[c * height * width + y * width + x]!;
                }
                image_data.data[pixel_idx + 3] = 255;
            }
        }
    }
    ctx.putImageData(image_data, 0, 0);
    return canvas;
}

export function ort_tensor_to_html_image(tensor: ort.Tensor): HTMLImageElement {
    const canvas = ort_tensor_to_html_canvas(tensor);
    return html_canvas_to_html_image(canvas);
}

export function ort_tensor_to_cv_mat(tensor: ort.Tensor): cv.Mat {
    const dims = tensor.dims;
    if (dims.length !== 4 || dims[0] !== 1 || dims[1] !== 3) {
        throw new Error("expected tensor shape [1, 3, H, W]");
    }
    const [, , height, width] = dims;
    const mat = new cv.Mat(height, width, cv.CV_32FC4);
    const tensor_data = tensor.data as Float32Array;
    const mat_data = mat.data as unknown as Float32Array;
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const pixel_idx = (y * width + x) * 4;
            for (let c = 0; c < 3; c++) {
                mat_data[pixel_idx + c] = tensor_data[c * height * width + y * width + x]!;
            }
            mat_data[pixel_idx + 3] = 1.0;
        }
    }
    return mat;
}

// TODO: from cv mat
// ============================================================================================
