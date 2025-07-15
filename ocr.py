import easyocr
import pytesseract
from util_functions import calculateTime
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont


def preprocess_image(image, resize=150):
    # Load the image
    if isinstance(image, str):
        image = cv2.imread(image)
    # Grayscale
    processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize
    if isinstance(resize, str) and resize == 'constant':
        processed = cv2.resize(processed, (700, 450), interpolation=cv2.INTER_CUBIC)
    else:
        scale_percent = resize
        width = int(processed.shape[1] * scale_percent / 100)
        height = int(processed.shape[0] * scale_percent / 100)
        processed = cv2.resize(processed, (width, height), interpolation=cv2.INTER_CUBIC)
    # Contrast
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    processed = clahe.apply(processed)
    # Denoise
    processed = cv2.fastNlMeansDenoising(processed, h=25)
    # Sharpness
    blurred = cv2.GaussianBlur(processed, (0, 0), sigmaX=3)
    processed = cv2.addWeighted(processed, 2, blurred, -0.5, 0)

    return processed





# ======================= Constants ======================= #
CIN_ALLOWED_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
NAME_ALLOWED_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ-\\ \\'"
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# ======================= Debugging ======================= #
def draw_bbox(image, bbox):
    cv2.rectangle(image, bbox[0], bbox[1], (0, 0, 255), 1)
    cv2.imshow("output", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def check_ocr():
    mistakes = {}

    for letter in ALPHABET:
        print("- "*30 + " Processing Letter: " + letter + " -"*30)
        results = process_image("./cins/", f'cin_{letter}.png')
        print(results)
        if (results[0][0] != letter):
            mistakes[letter] = results[0][0]

    return mistakes




# ======================= Main Program ======================= #
def read_text_from_image(image, mode="bbox", ALLOWED_CHARS="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789\\ \\'"):

    if mode == "text" :
        if isinstance(image, str):
            img = Image.open(image)
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        else:
            raise TypeError("Unsupported image type")
        config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=' + ALLOWED_CHARS
        results = pytesseract.image_to_string(img, lang='eng+fra', config=config)
        return [results]


    img = cv2.imread(image)
    # h, w = img.shape[:2]
    # config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=' + ALLOWED_CHARS
    # boxes = pytesseract.image_to_boxes(img, config=config, lang='eng')

    # boxes = boxes.splitlines()
    # borders = [[w,h], [0, 0]]

    # print(boxes)
    # for b in boxes:
    #     b = b.split()
    #     x1, y1, x2, y2 = int(b[1]), int(b[2]), int(b[3]), int(b[4])
    #     y1, y2 = h - y1, h - y2
    #     borders[0][0] = min(borders[0][0], x1)
    #     borders[0][1] = min(borders[0][1], y2)
    #     borders[1][0] = max(borders[1][0], x2)
    #     borders[1][1] = max(borders[1][1], y1)

    reader = easyocr.Reader(['fr', 'en'], gpu=True)
    results = reader.readtext(image, detail=1, allowlist=ALLOWED_CHARS)

    borders = [results[0][0][0], results[0][0][2]]
    # draw_bbox(img, borders)
    # print(borders)

    return borders


def draw_letter(text, folder, img, bbox):
    if isinstance(img, str):
        image = Image.open(folder + img)
    else:
        image = img

    draw = ImageDraw.Draw(image)
    font_size = (bbox[1][1] - bbox[0][1]) * 0.9
    width = font_size/1.75
    text_position = int(bbox[0][0] - width), int(bbox[0][1] * 0.95)


    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    draw.text(text_position, text, fill='black', font=font)


    image.save(folder + "corrected_" + img)


def process_image(folder, path, correcting=False):
    calculateTime()
    if correcting:
        # Preprocessing
        processed_image = preprocess_image(folder + path)

        draw_bbox(processed_image, np.array([[0,0],[1000,1000]], dtype=np.int32))

        processed_image = Image.fromarray(processed_image)
        processed_image.save(folder + "processed_" + path, dpi=(300, 300))

        # Collect Bounding Box
        bbox = read_text_from_image(folder + "processed_" + path, ALLOWED_CHARS=CIN_ALLOWED_CHARS)

        # Add Letter
        # draw_bbox(folder + path, bbox)
        draw_letter("A", folder, "processed_" + path, bbox)


        text = "Saved at " + folder + "corrected_processed_" + path

    if not correcting:

        # Perform OCR
        data = read_text_from_image(folder + "corrected_processed_" + path, mode="text", ALLOWED_CHARS=CIN_ALLOWED_CHARS)
        print(data)
        for i, result in enumerate(data):
            result = result[1:]
            if result[0] == '0' :
                result = "O" + result[1:]
            elif result[1] == '0' and result[0] == 'D':
                result = "undefined"
            data[i] = result
        text = data

    print(f"Processing in {calculateTime():.2f}")
    return text






# ======================= Test Program ======================= #
if __name__ == "__main__":

    # imgs = ["cropped-2.png"]
    imgs = [f"cin{i}.png" for i in range(1,8)]
    # imgs = [f'cin_{letter}.png' for letter in ALPHABET]
    # imgs = [f'test{i}.png' for i in range(1,9)]
    # imgs = ["cin2.png", "cin-1.png"]

    folder ="./confidentiels/"

    for img in imgs:
        process_image(folder, img, correcting=True)
        results = process_image(folder, img)
        print("\n".join(results))
