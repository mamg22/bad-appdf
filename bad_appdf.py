import cv2 as cv
import numpy as np
import pymupdf as pm


PLAY_SCRIPT = """
function play() {
    if (this.pageNum < this.numPages - 1) {
        this.pageNum += 1;
    }
    else {
        app.clearInterval(interval);
    }
}
interval = app.setInterval('play()', 1000/10);
"""


def main():
    doc = pm.Document()

    first = doc.new_page()

    wig = pm.Widget()
    wig.field_type = pm.PDF_WIDGET_TYPE_BUTTON
    wig.field_name = "Button"
    wig.field_label = "Button"
    wig.button_caption = "Play"
    wig.fill_color = (0.0, 0.7, 0.7)
    wig.rect = pm.Rect(10, 10, 100, 30)
    wig.script = PLAY_SCRIPT

    first.add_widget(wig)

    for (contours, hierarchy), dimensions in iter_frames():
        p = doc.new_page(width=dimensions[0], height=dimensions[1])
        if not contours:
            continue

        for n, contour in enumerate(contours):
            if hierarchy[0][n][-1] >= 0:
                color = (1, 1, 1)
            else:
                color = (0, 0, 0)
            contour = np.vstack(contour).squeeze().tolist()
            if not isinstance(contour[0], list):
                contour = [contour]
            p.draw_polyline(contour, fill=color)

    doc.ez_save("bad.pdf")


def iter_frames():
    vid = cv.VideoCapture("bad_apple_360p.mp4")

    width = vid.get(cv.CAP_PROP_FRAME_WIDTH)
    height = vid.get(cv.CAP_PROP_FRAME_HEIGHT)
    frame_count = int(vid.get(cv.CAP_PROP_FRAME_COUNT))

    while True:
        pos = int(vid.get(cv.CAP_PROP_POS_FRAMES))
        ret, frame = vid.read()
        if not ret:
            return
        if pos % 3 != 0:
            continue

        print(f"{pos: 5d}/{frame_count}")

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV)

        contours, hierarchy = cv.findContours(thresh, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

        yield (contours, hierarchy), (width, height)


if __name__ == "__main__":
    main()
