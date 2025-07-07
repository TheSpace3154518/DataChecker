from io import TextIOBase
import easyocr
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageFont, ImageDraw
import numpy as np
import cv2

# ======================= Variables ======================= #
ALLOWED_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"



def preprocess_image(img):

    image = Image.open(img)

    # resizing
    factor = 1
    processed = image.resize((int(image.width * factor), int(image.height)), Image.LANCZOS)

    #Grayscale
    processed = processed.convert("L")

    #Median Filter
    # processed = processed.filter(ImageFilter.MinFilter(size=3))

    #Contrast
    enhancer = ImageEnhance.Contrast(processed)
    processed = enhancer.enhance(1.5)

    # Brightness
    # enhancer = ImageEnhance.Sharpness(image)
    # image = enhancer.enhance(2)


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
    image = Image.open(folder + "processed_" + img)

    draw = ImageDraw.Draw(image)

    font_size = (bbox[2][1] - bbox[0][1])
    width = font_size/2
    text_position = bbox[0][0] - width, bbox[0][1]

    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    draw.text(text_position, text, fill='black', font=font)
    # image.show()
    image.save(folder + "corrected_" + img)



def draw_bbox(img, bbox):

    image = Image.open(img)
    draw = ImageDraw.Draw(image)

    # Define box (left, top, right, bottom)
    box = (bbox[0][0], bbox[0][1], bbox[2][0], bbox[2][1])
    draw.rectangle(box, outline='red', width=1)
    image.show()


def process_image(folder, path, correcting=False):
    if correcting:
        # Pre-Processing
        processed_image = preprocess_image(folder + path)
        processed_image.save(folder + "processed_" + path, dpi=(300, 300))

        # Perform OCR
        data = read_text_from_image(folder + "processed_" + path, 6)

        for res in data :
            # draw_bbox(folder + path, res[0])
            draw_letter("A", folder, path, res[0])

    if not correcting:
        # Perform OCR
        data = read_text_from_image(folder + "corrected_" + path, 6)

    text = [f"{result[1]} | conf : {result[2]:.2f}" for result in data]
    return text

if __name__ == "__main__":

    imgs = ["cropped-1.png", "cropped-2.png"]
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
