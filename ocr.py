import easyocr
import pytesseract
from util_functions import calculateTime
from Preprocessing import ImageProcessor
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

    if isinstance(image, str):
        img = cv2.imread(image)
    else:
        img = image

    if mode == "text" :
        config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=' + ALLOWED_CHARS
        results = pytesseract.image_to_string(img, lang='eng+fra+ara', config=config)
        return [results]

    reader = easyocr.Reader(lang_list=['fr', 'en'], gpu=True)
    results = reader.readtext(img, detail=1, allowlist=ALLOWED_CHARS)

    borders = [results[0][0][0], results[0][0][2]]
    return borders


def draw_letter(text, img, bbox):
    if isinstance(img, str):
        image = Image.open(img)
    else:
        image = Image.fromarray(img)

    draw = ImageDraw.Draw(image)
    font_size = (bbox[1][1] - bbox[0][1]) * 0.9
    width = font_size/1.6
    text_position = int(bbox[0][0] - width), bbox[0][1]

    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    draw.text(text_position, text, fill='black', font=font)

    cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    cv2.imshow("Letter", cv_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return cv_img


def process_image(image, correcting=False):
    calculateTime()
    if isinstance(image, str):
        image = cv2.imread(image)

    if correcting:

        # Preprocessing
        processor = ImageProcessor(
                resize="constant:800",
                grayscale=True,
                contrast_clip_limit=2,
                denoise_h=25,
                sharpness_alpha=2,
                sharpness_beta=0.5
            )
        processed_image = processor.preprocess_image(image)



        # Collect Bounding Box
        bbox = read_text_from_image(processed_image, ALLOWED_CHARS=CIN_ALLOWED_CHARS)
        draw_bbox(processed_image.copy(), bbox)

        # Add Letter
        corrected = draw_letter("A", processed_image, bbox)

        return corrected

    if not correcting:

        # Perform OCR
        data = read_text_from_image(image, mode="text", ALLOWED_CHARS=CIN_ALLOWED_CHARS)
        data = data[0].split("\n")
        print(data)
        for i, result in enumerate(data):
            if result != '':
                result = result[1:]
                if result[0] == '0' :
                    result = "O" + result[1:]
                elif len(result) >= 2 and result[1] == '0' and result[0] == 'D':
                    result = "undefined"
                data[i] = result
        return data







# ======================= Test Program ======================= #
if __name__ == "__main__":

    imgs = ["cropped-2.png"]
    # imgs = [f"cin{i}.png" for i in range(1,8)]
    # imgs = [f'cin_{letter}.png' for letter in ALPHABET]
    # imgs = [f'test{i}.png' for i in range(1,9)]
    # imgs = ["cin2.png", "cin-1.png"]

    folder ="./Recruits/"

    for img in imgs:
        process_image(folder + img, correcting=True)
        results = process_image(folder + img)
        print("\n".join(results))
