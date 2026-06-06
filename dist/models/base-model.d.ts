import * as ort from "onnxruntime-web/all";
export interface BaseModel {
    ort_session: ort.InferenceSession | null;
    create_session(model_path: Uint8Array, options: ort.InferenceSession.SessionOptions): Promise<void>;
    preprocess(input: HTMLCanvasElement, width: number, height: number): ort.Tensor;
    run_inference(input_tensors: Array<ort.Tensor>): Promise<ort.Tensor>;
    postprocess(input: ort.Tensor, width: number, height: number): HTMLCanvasElement;
}
//# sourceMappingURL=base-model.d.ts.map