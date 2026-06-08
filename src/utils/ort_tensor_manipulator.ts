import * as ort from "onnxruntime-web/all";
import * as cv from "@techstark/opencv-js";

export function RGBA_to_RGB(input: ort.Tensor, width: number, height: number): ort.Tensor {
    let rgb_array: Float32Array = new Float32Array(3 * width * height);

    for (let i: number = 0; i < width * height; i++){
        rgb_array[0 * width * height + i] = input.data[0 * width * height + i] as number;
        rgb_array[1 * width * height + i] = input.data[1 * width * height + i] as number;
        rgb_array[2 * width * height + i] = input.data[2 * width * height + i] as number;
    }

    return new ort.Tensor("float32", rgb_array, [1, 3, width, height]);
}

export function RGB_to_RGBA(input: ort.Tensor, width: number, height: number): ort.Tensor {
    let rgb_array: Float32Array = new Float32Array(4 * width * height);

    for (let i: number = 0; i < width * height; i++){
        rgb_array[0 * width * height + i] = input.data[0 * width * height + i] as number;
        rgb_array[1 * width * height + i] = input.data[1 * width * height + i] as number;
        rgb_array[2 * width * height + i] = input.data[2 * width * height + i] as number;
        rgb_array[3 * width * height + i] = 0;

        if (input.data[0 * width * height + i] as number != 0 ||
            input.data[1 * width * height + i] as number != 0 ||
            input.data[2 * width * height + i] as number != 0
        ) {
            rgb_array[3 * width * height + i] = 255;
        }
    }

    return new ort.Tensor("float32", rgb_array, [1, 4, width, height]);
}