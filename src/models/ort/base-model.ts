import * as ort from "onnxruntime-web";

export interface BaseModel {
    ort_session: ort.InferenceSession | null;

    create_session(model_path: Uint8Array, options: ort.InferenceSession.SessionOptions): Promise<void>;

    // TODO: instead of array, better make them hashmap.
    preprocess(input: Array<HTMLCanvasElement>, width: number, height: number): Array<ort.Tensor>;

    // TODO: the input & output nodes must be arguments too.
    run_inference(input_tensors: Array<ort.Tensor>): Promise<ort.Tensor>;

    // TODO: instead of array, better make them hashmap.
    postprocess(input: Array<ort.Tensor>, width: number, height: number): HTMLCanvasElement;
}