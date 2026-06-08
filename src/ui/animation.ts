export class Animation {
    public add_cursor_hover_effect(div_elem: HTMLDivElement, image_elem: HTMLImageElement, translation_factor: number): HTMLImageElement {
        div_elem.addEventListener("mousemove", (event) => {
            image_elem.style.transition = "none";
            this.updateParallax(div_elem, image_elem, event.clientX, event.clientY, translation_factor);
        });

        div_elem.addEventListener("mouseleave", () => {
            image_elem.style.transition = "transform 0.5s cubic-bezier(0.25, 0.8, 0.25, 1)";
            image_elem.style.transform = "translateX(0) translateY(0) scale(0.9)";
        });

        return image_elem;
    }

    public add_gyroscope_effect(){
        // TODO
    }

    public add_rubber_banding_effect(){
        // TODO
    }

    private updateParallax(div_elem: HTMLDivElement, img_elem: HTMLImageElement, cursor_x: number, cursor_y: number, translation_factor: number) {
        if (!div_elem) return;

        const div_elem_rect = div_elem.getBoundingClientRect();
        const x_center = div_elem_rect.left + div_elem_rect.width / 2;
        const distance_from_center_x = cursor_x - x_center;

        const y_center= div_elem_rect.top + div_elem_rect.height / 2;
        const distance_from_center_y = cursor_y - y_center;

        const translation_x = distance_from_center_x * translation_factor;
        const translation_y = distance_from_center_y* translation_factor;
        img_elem.style.transform = `translateX(${translation_x}px) translateY(${translation_y}px) scale(0.9)`;
    }
}