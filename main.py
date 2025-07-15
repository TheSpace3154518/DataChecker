from warp import warp_img
from crop import crop_img
import cv2


def read_img(folder, img):
    stretched = warp_img(folder,img)
    cv2.imwrite(folder + "output_" + img, stretched)
    crop_img(folder, "output_" + img)


if __name__ == "__main__":
    folder="./confidentiels/"
    imgs = [f"output-{i}.png" for i in range(2, 3)]
    for img in imgs:
        read_img(folder, img)
