from warp import warp_img
from crop import crop_img
from pdf2image import convert_from_path
import cv2
import numpy as np
import os


def read_img(img):
    stretched = warp_img(img)
    # crop_img(stretched)


if __name__ == "__main__":
    folder="./confidentiels/"

    # imgs = [f"output-{i}.png" for i in range(2, 3)]
    # imgs = ["cin-1.png", "cin2.png"]
    # for img in imgs:
    #     read_img(folder + img)

    # pdf_files = [f for f in os.listdir(folder) if f.lower().endswith('.pdf')]
    pdf_files = [f"cin{i}.pdf" for i in range(1, 8)]

    for pdf in pdf_files:
        print("-"*30 + f" Processing {pdf} " + "-"*30)
        images = convert_from_path(folder + pdf, dpi=300)
        for i, image in enumerate(images):
            cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            read_img(cv_img)
