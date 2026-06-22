import * as ModelLib from "./lib.ts";

export class Model {
    // fields
    model_name: String;
    execution_provider: String;

    // public functions
    constructor(model_name: String, execution_provider: String){
        this.model_name = model_name,
        this.execution_provider = execution_provider;
    }

    public async load_model(){
        // TODO
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