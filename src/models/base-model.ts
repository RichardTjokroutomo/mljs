import * as ort from "onnxruntime-web/all";
import * as cv from "@techstark/opencv-js";

export interface BaseModel {
    ort_session: ort.InferenceSession | null;

    create_session(model_path: Uint8Array, options: ort.InferenceSession.SessionOptions): Promise<void>;
    preprocess(input: HTMLCanvasElement, width: number, height: number): ort.Tensor;

    // TODO: the input & output nodes must be arguments too.
    run_inference(input_tensors: Array<ort.Tensor>): Promise<ort.Tensor>;
    postprocess(input: ort.Tensor, width: number, height: number): HTMLCanvasElement;
}