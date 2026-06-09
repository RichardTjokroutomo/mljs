import cvModule from "@techstark/opencv-js";

const cv = (cvModule as any).default ?? cvModule; // Handle both default and named exports from OpenCV.js

export function resize_html_canvas(canvas: HTMLCanvasElement, width: number, height: number): HTMLCanvasElement {
    const src = cv.imread(canvas);
    const dst = new cv.Mat();
    cv.resize(src, dst, new cv.Size(width, height), 0, 0, cv.INTER_LINEAR);
    src.delete();

    const out = document.createElement("canvas");
    out.width = width;
    out.height = height;
    cv.imshow(out, dst);
    dst.delete();

    return out;
}
