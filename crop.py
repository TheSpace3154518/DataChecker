from ocr import read_text_from_image, process_image, NAME_ALLOWED_CHARS, preprocess_image
from PIL import Image
import cv2
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
    image = preprocess_image(image)
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
    image = preprocess_image(image)
    cv2.imshow("output", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    results = read_text_from_image(image, mode="text", ALLOWED_CHARS=NAME_ALLOWED_CHARS)
    texts = [result.strip() for result in results[0].split("\n") if result.strip()]

    return valid, texts[:2]

def crop_img(folder, img):
    image = cv2.imread(folder + img)
    valid, texts = check_model(image)

    if not valid:
        raise ValueError("Not CIN")

    print(texts)

    if len(texts) == 0:

        # Update Texts
        orig = image.copy()
        H, W = image.shape[:2]
        x1, y1, x2, y2 = proportion(9/25, 1/5, 1/2, 1/3, W, H)
        orig = orig[y1:y2, x1:x2]
        draw_bbox(image, np.array([[x1,y1],[x2,y2]], dtype=np.int32))

        orig = preprocess_image(orig)
        cv2.imshow("output", orig)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        results = read_text_from_image(orig, mode="text", ALLOWED_CHARS=NAME_ALLOWED_CHARS)
        texts = [result.strip() for result in results[0].split("\n") if result.strip()]
        texts = texts[:2]

        # Define CIN Zone

        orig = image.copy()
        H, W = image.shape[:2]
        x1, y1, x2, y2 = proportion(9/95, 6/7, 3/14, 1, W, H)
        draw_bbox(orig, np.array([[x1,y1],[x2,y2]], dtype=np.int32))
    else :
        orig = image.copy()
        H, W = image.shape[:2]
        x1, y1, x2, y2 = proportion(2/3, 3/4, 1/2, 1/2, W, H)
        draw_bbox(image, np.array([[x1,y1],[x2,y2]], dtype=np.int32))

    orig = orig[y1:y2, x1:x2]
    cv2.imwrite(folder + "cropped_" + img, orig)

    process_image(folder, "cropped_" + img, correcting=True)
    results = process_image(folder, "cropped_" + img)
    print("\n".join(texts))
    print("\n".join(results))


if __name__ == "__main__":
    folder = "./Recruits/"
    img = "aspct.png"
    crop_img(folder, img)
