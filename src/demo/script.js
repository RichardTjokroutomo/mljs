import * as mljs from "../../dist/index.js";

console.log(mljs);
console.log(Object.keys(mljs));

let spatial_scene = new mljs.SpatialScene();

console.log("SpatialScene initialized:", spatial_scene);

await spatial_scene.initialize_sessions("../../model_binaries/depth_anything_v2_vits_quantized.onnx", "../../model_binaries/migan_processed.onnx");

console.log("Sessions initialized!");

document.getElementById("process").addEventListener("click", () => {
    console.log("===============================")
    spatial_scene.convert_single_image(document.getElementById("result"), 4);
});