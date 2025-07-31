from warp import warp_img
from crop import crop_img
from ocr import read_text_from_image
from pdf2image import convert_from_path
import cv2
from Preprocessing import ImageProcessor
import numpy as np
from smart import apply_smart_crop
import re



# Preprocessing l OCR
# Preprocessing Carte nationale Checker



def read_img(img):
    stretched = warp_img(img)
    crop_img(stretched)


if __name__ == "__main__":
    folder="./confidentiels/"

    # ======= Default Method ==========

    # imgs = [f"output-{i}.png" for i in range(2, 3)]
    # imgs = ["rotated_270.jpeg", "rotated_m9loba.jpeg","rotated_shwiya.jpeg"]
    # for img in imgs:
    #     cv_img = cv2.imread(folder + img)
    #     read_img(cv_img)

    # pdf_files = [f for f in os.listdir(folder) if f.lower().endswith('.pdf')]
    # pdf_files = [f"cin{i}.pdf" for i in range(14, 18)]

    # for pdf in pdf_files:
    #     print("-"*30 + f" Processing {pdf} " + "-"*30)
    #     images = convert_from_path(folder + pdf, dpi=300)
    #     for i, image in enumerate(images):
    #         cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    #         read_img(cv_img)



    # =========== Including the 2 methods ===========

    pdf_files = [f"cin{i}.pdf" for i in range(1, 19)]

    for pdf in pdf_files:
        print("-"*30 + f" Processing {pdf} " + "-"*30)
        images = convert_from_path(folder + pdf, dpi=300)
        for image in images:
            cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            processor = ImageProcessor(
                    grayscale=True,
                    contrast_clip_limit=1,
                    denoise_h=25,
                    sharpness_alpha=2.5,
                    sharpness_beta=0.5
                )
            image = processor.preprocess_image(cv_img)

            results = read_text_from_image(image, mode="text")

            pattern = r'[A-Z]{1,2}[0-9]{3,8}$'
            cins = []
            for result in results :
                split_result = re.split(r'[\n\ ]', result)
                split_result = [res.strip() for res in split_result]
                cins.extend(split_result)

            # print("\n".join(cins[-30:]))
            cins = [cin for cin in cins if bool(re.match(pattern, cin))]

            print("Verdict : ")
            print("\n".join(cins))
            if len(cins) == 0:
                read_img(cv_img)

            # cv2.imshow("Processed Image", image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
