from io import TextIOBase
import easyocr
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageFont, ImageDraw
import numpy as np
import cv2

# ======================= Variables ======================= #
ALLOWED_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"



def preprocess_image(img):

    # Load the image
    image = cv2.imread(img)

    # Grayscale
    processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize
    scale_percent = 150  # scale by 200%
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
    processed = cv2.addWeighted(processed, 2.5, blurred, -0.5, 0)

    return processed


def read_text_from_image(image, val):

    # img = Image.open(image)
    # config = r'--oem 3 --psm ' + str(val)
    # results = pytesseract.image_to_string(img, lang='eng+fr', config=config)

    reader = easyocr.Reader(['fr', 'en'], gpu=True)
    results = reader.readtext(image, detail=1, allowlist=ALLOWED_CHARS)

    return results


alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
def check_ocr():
    mistakes = {}

    for letter in alphabet:
        print("- "*30 + " Processing Letter: " + letter + " -"*30)
        results = process_image("./cins/", f'cin_{letter}.png')
        print(results)
        if (results[0][0] != letter):
            mistakes[letter] = results[0][0]

    return mistakes

def draw_letter(text, folder, img, bbox):
    image = Image.open(folder + img)

    draw = ImageDraw.Draw(image)

    font_size = (bbox[2][1] - bbox[0][1]) * 0.9
    width = font_size/1.75
    text_position = min(bbox[0][0],bbox[1][0],bbox[2][0],bbox[3][0]) - width, min(bbox[0][1],bbox[1][1],bbox[2][1],bbox[3][1])

    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    draw.text(text_position, text, fill='black', font=font)
    # image.show()
    image.save(folder + "corrected_" + img)



def draw_bbox(img, bbox):

    image = Image.open(img)
    draw = ImageDraw.Draw(image)

    # Define box (left, top, right, bottom)
    box = (bbox[0][0], bbox[0][1], bbox[2][0], bbox[2][1])
    draw.rectangle(box, outline='red', width=2)
    image.show()


def process_image(folder, path, correcting=False):
    if correcting:
        # Preprocessing
        processed_image = preprocess_image(folder + path)
        processed_image = Image.fromarray(processed_image)
        processed_image.save(folder + "processed_" + path, dpi=(300, 300))

        # Perform OCR
        data = read_text_from_image(folder + "processed_" + path, 6)

        for res in data :
            # draw_bbox(folder + path, res[0])
            draw_letter("A", folder, "processed_" + path, res[0])

    if not correcting:
        # processed_image = preprocess_image(folder + path)
        # processed_image = Image.fromarray(processed_image)
        # processed_image.save(folder + "processed_" + path, dpi=(300, 300))

        # Perform OCR
        data = read_text_from_image(folder + "corrected_processed_" + path, 6)

    text = [f"{result[1]} | conf : {result[2]:.2f}" for result in data]
    return text

if __name__ == "__main__":

    imgs = ["cropped-2.png", "test_cin2.png"]
    # imgs = [f'cin_{letter}.png' for letter in alphabet]
    # imgs = [f'test{i}.png' for i in range(1,9)]
    # imgs = ["cin2.png", "cin-1.png"]

    folder ="./Recruits/"

    # Take 1
    for img in imgs:

        # Add Correction
        print("-"*30 + " Correcting Image: " + img + "-"*30)
        results = process_image(folder, img, correcting=True)
        print("\n".join(results))

        # Attempt to read
        print("-"*30 + " Processing Image: " + img + "-"*30)
        results = process_image(folder, img)
        print("\n".join(results))
