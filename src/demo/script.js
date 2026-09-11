import * as mljs from "../../dist/index.js";

console.log(mljs);
console.log(Object.keys(mljs));

let spatial_scene = new mljs.SpatialScene();

console.log("SpatialScene initialized:", spatial_scene);

await spatial_scene.initialize_sessions("../../model_binaries/webnn/depth-anything-v2/depth_anything_v2_quantized.js", "../../model_binaries/webnn/migan/migan.js");

console.log("Sessions initialized!");

document.getElementById("process").addEventListener("click", () => {
    console.log("===============================");
    spatial_scene.convert_single_image(document.getElementById("result"), 4);
});