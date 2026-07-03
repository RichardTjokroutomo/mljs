import * as ModelLib from "./lib.ts";
import * as ort from "onnxruntime-web";

export class Model {
    // fields
    model_name: String;
    execution_provider: String;
    model: ort.InferenceSession | null = null;

    // public functions
    constructor(model_name: String, execution_provider: String){
        // TODO: check if the arguments have valid values using enum.
        this.model_name = model_name,
        this.execution_provider = execution_provider;
    }

    public async load_model(model_path: Uint8Array){
        switch (this.execution_provider) {
            case 'webnn':
                break;

            case 'ort':
                break;
        }
    }

    public preprocess(){
        // TODO
    }

    public async run_inference(){
        // TODO
    }

    public postprocess(){
        // TODO
    }

    // private functions
}