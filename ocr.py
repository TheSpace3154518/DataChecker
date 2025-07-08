import easyocr
import time
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# ======================= TODO ======================= #
# Check Doc
# Crop
# Refactoring



# ======================= Constants ======================= #
ALLOWED_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '"
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# ======================= Debugging ======================= #
def draw_bbox(img, bbox):
    image = Image.open(img)
    draw = ImageDraw.Draw(image)
    box = (bbox[0][0], bbox[0][1], bbox[2][0], bbox[2][1])
    draw.rectangle(box, outline='red', width=2)
    image.show()

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
def preprocess_image(img):

    # Load the image
    image = cv2.imread(img)

    # Grayscale
    processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize
    scale_percent = 100
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


def read_text_from_image(image, detail):

    reader = easyocr.Reader(['en'], gpu=True)
    results = reader.readtext(image, detail=detail, allowlist=ALLOWED_CHARS)

    return results


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


def process_image(folder, path, correcting=False):
    if correcting:
        # Preprocessing
        processed_image = preprocess_image(folder + path)
        cv2.imwrite(folder + "processed_" + img, processed_image)

        # Correct Image by adding letter
        data = read_text_from_image(folder + "processed_" + path, 1)

        for res in data :
            # draw_bbox(folder + path, res[0])
            draw_letter("A", folder, "processed_" + path, res[0])

        text = [result[1] for result in data]

    if not correcting:

        # Perform OCR
        data = read_text_from_image(folder + "corrected_processed_" + path, 0)
        for i, result in enumerate(data):
            result = result[1:]
            if result[0] == '0' :
                result = "O" + result[1:]
            elif result[1] == '0' and result[0] == 'D':
                result = "undefined"
            data[i] = result
        text = data


    return text


# ======================= Test Program ======================= #
if __name__ == "__main__":

    # imgs = ["cropped-2.png", "test_cin2.png"]
    imgs = [f'cin_{letter}.png' for letter in ALPHABET]
    # imgs = [f'test{i}.png' for i in range(1,9)]
    # imgs = ["cin2.png", "cin-1.png"]

    folder ="./cins/"

    for img in imgs:

        start_time = time.time()
        print("-"*30 + " Processing Image: " + img + "-"*30)
        # Add Correction
        # print("-"*30 + " Correcting Image: " + img + "-"*30)
        results = process_image(folder, img, correcting=True)
        # print(f"Saved at {folder + 'corrected_processed_' + img}")

        step1 = time.time()
        correction_time = step1 - start_time

        # Attempt to read
        results = process_image(folder, img)

        end_time = time.time()
        read_time = end_time - step1
        execution_time = end_time - start_time

        print(f"Corrected in {correction_time:.2f} s")
        print(f"Read in {read_time:.2f} s")
        print(f"Total in {execution_time:.2f} s")
        print("\n".join(results))
