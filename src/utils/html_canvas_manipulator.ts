import cvModule from "@techstark/opencv-js";

const cv = (cvModule as any).default ?? cvModule; // Handle both default and named exports from OpenCV.js

export function resize_canvas_native(canvas: HTMLCanvasElement, width: number, height: number): HTMLCanvasElement {
    const out = document.createElement("canvas");
    out.width = width;
    out.height = height;
    const ctx = out.getContext("2d")!;
    ctx.imageSmoothingEnabled = false; // Match INTER_NEAREST behavior
    ctx.drawImage(canvas, 0, 0, width, height);
    return out;
}

export function resize_html_canvas(canvas: HTMLCanvasElement, width: number, height: number): HTMLCanvasElement {
    const src = cv.imread(canvas);
    const dst = new cv.Mat();
    cv.resize(src, dst, new cv.Size(width, height), 2, 2, cv.INTER_LANCZOS4);
    src.delete();

    const out = document.createElement("canvas");
    out.width = width;
    out.height = height;
    cv.imshow(out, dst);
    dst.delete();

    return out;
}
