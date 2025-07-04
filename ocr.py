import easyocr
from PIL import Image, ImageEnhance
import numpy as np

def preprocess_image(image):

    image = Image.open(image)

    # resizing
    image = image.resize((image.width * 2, image.height * 2), Image.LANCZOS)

    #Grayscale
    image = image.convert("L")

    #Contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)

    # Brightness
    # enhancer = ImageEnhance.Sharpness(image)


    return image






def read_text_from_image(image):

    reader = easyocr.Reader(['en', 'fr'], gpu=False)
    results = reader.readtext(image, detail=1, contrast_ths=0.05)
    results = [f"{result[1]} | conf : {result[2]:.2f}" for result in results]
    return results



if __name__ == "__main__":

    imgs = ["cin-1.png"]
    folder ="./Recruits/"
    for img in imgs:
        processed_image = preprocess_image(folder + img)
        processed_image.save("processed_" + img)
        processed_image.show()

        text = read_text_from_image("processed_" + img)
        print("\n".join(text))
