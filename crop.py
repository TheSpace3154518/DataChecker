from ocr import read_text_from_image, process_image, NAME_ALLOWED_CHARS
from Preprocessing import ImageProcessor
from test import detect_text_presence
import cv2
from util_functions import drawContours
import numpy as np


def draw_bbox(image, bbox):
    cv2.rectangle(image, bbox[0], bbox[1], (0, 0, 255), 1)
    cv2.imshow("output", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def proportion(r1, r2, r3, r4, W, H):
    x1, y1 = W * r1, H * r2
    x2, y2 = x1 + (W - x1) * r3, y1 + (H - y1) * r4
    return int(x1), int(y1), int(x2), int(y2)


def check_model(orig):

    image = orig.copy()
    H, W = image.shape[:2]

    # Crop Check Zone
    x1, y1, x2, y2 = proportion(0, 0, 3/8, 1/5, W, H)
    draw_bbox(image, np.array([[x1,y1],[x2,y2]], dtype=np.int32))

    image = image[y1:y2, x1:x2]
    draw_bbox(image, np.array([[0,0],[image.shape[1],image.shape[0]]], dtype=np.int32))

    # Read From Check Zone

    processor = ImageProcessor(
            resize="constant:800",
            grayscale=True,
            contrast_clip_limit=2,
            denoise_h=25,
            sharpness_alpha=1.5,
            sharpness_beta=0.5
        )
    image = processor.preprocess_image(image)
    cv2.imshow("output", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    results = read_text_from_image(image, mode="text", ALLOWED_CHARS=NAME_ALLOWED_CHARS)
    texts = results[0].split("\n")

    valid = False
    for text in texts:
        if "CARTE NATIONALE D" in text.strip():
            valid = True


    if not valid:
        return False, []


    # Check Model Type
    image = orig.copy()
    x1, y1, x2, y2 = proportion(0, 1/4, 3/8, 5/12, W, H)
    draw_bbox(image, np.array([[x1,y1],[x2,y2]], dtype=np.int32))

    image = image[y1:y2, x1:x2]

    text_presence, rects = detect_text_presence(image)

    drawContours(image, rects)

    return valid, text_presence

def crop_img(img):
    if isinstance(img, str):
        image = cv2.imread(img)
    else:
        image = img.copy()
    valid, kayn_text = check_model(image)

    if not valid:
        print("Not a CIN")
        return


    if not kayn_text:

        # Update Texts
        orig = image.copy()
        H, W = image.shape[:2]
        x1, y1, x2, y2 = proportion(9/25, 1/5, 1/2, 1/3, W, H)
        orig = orig[y1:y2, x1:x2]
        draw_bbox(image, np.array([[x1,y1],[x2,y2]], dtype=np.int32))


        processor = ImageProcessor(
                resize="constant:800",
                grayscale=True,
                contrast_clip_limit=2,
                denoise_h=25,
                sharpness_alpha=2.5,
                sharpness_beta=0.5
            )
        orig = processor.preprocess_image(orig)
        cv2.imshow("output", orig)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        results = read_text_from_image(orig, mode="text", ALLOWED_CHARS=NAME_ALLOWED_CHARS)
        texts = [result.strip() for result in results[0].split("\n") if result.strip()]
        texts = texts[:2]

        # Define CIN Zone

        orig = image.copy()
        H, W = image.shape[:2]
        x1, y1, x2, y2 = proportion(8/95, 5.5/7, 3/14, 1, W, H)
        # x1, y1, x2, y2 = (0, 0, W, H)
        draw_bbox(orig, np.array([[x1,y1],[x2,y2]], dtype=np.int32))

    else :

        # Update Texts
        orig = image.copy()
        H, W = image.shape[:2]
        x1, y1, x2, y2 = proportion(0, 1/4, 3/8, 5/12, W, H)
        orig = orig[y1:y2, x1:x2]
        draw_bbox(image, np.array([[x1,y1],[x2,y2]], dtype=np.int32))


        processor = ImageProcessor(
                resize="constant:800",
                grayscale=True,
                contrast_clip_limit=2,
                denoise_h=25,
                sharpness_alpha=2.5,
                sharpness_beta=0.5
            )
        orig = processor.preprocess_image(orig)
        cv2.imshow("output", orig)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        results = read_text_from_image(orig, mode="text", ALLOWED_CHARS=NAME_ALLOWED_CHARS)
        print(results)
        texts = [result.strip() for result in results[0].split("\n") if result.strip()]
        texts = texts[:2]

        # Define CIN Zone
        #
        orig = image.copy()
        H, W = image.shape[:2]
        x1, y1, x2, y2 = proportion(2.1/3, 2.7/4, 1.5/2, 1, W, H)
        # x1, y1, x2, y2 = (0, 0, W, H)
        draw_bbox(image, np.array([[x1,y1],[x2,y2]], dtype=np.int32))


    # Read CIN
    orig = orig[y1:y2, x1:x2]
    corrected = process_image(orig, correcting=True)
    results = process_image(corrected)


    print("\n".join(texts))
    print("\n".join(results))


if __name__ == "__main__":
    folder = "./confidentiels/"
    imgs = [f"cin-{i}.png" for i in range(1, 10)]
    for img in imgs:
        crop_img(folder + img)
