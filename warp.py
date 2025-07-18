import cv2
import numpy as np
from torch import imag
from crop import crop_img
from util_functions import drawContours
from Preprocessing import ImageProcessor
from test import detect_text_presence, adjust_gamma
from PIL import Image
from rembg import remove
import pytesseract


aspect_ratio = 1.554123711

# ======================= TODO ======================= #
# Read CIN                        - Done
# Add Letter                      - Done
# Optimize                        - Done

# Preprocess Image                - Done
# Find Contours                   - Done
# Detect Card                     - Done
# determine Orientation           - Done
# Warp card                       - Done
# Card Crop                       - Done
# Card Type                       - Done
# Data Crop                       - Done
# Optimize                        - Done

# bbox problem                    - Done
# borders dakhlin                 - Done
# Refactoring                     - Done
# pdf2img                         - Done

# Text Orientation                - Not yet
# Camscanner                      - Not yet
# EAST                            - Not yet




def detect_orientation(image):
    image = preprocess_image(image)
    img = Image.fromarray(image)
    config = r'--oem 3 --psm 6'
    data = pytesseract.image_to_data(img, config=config)
    print(data)
    for i in range(len(data['text'])):
        text = data['text'][i]
        conf = data['conf'][i]
        if text.strip():
            print(f"Word: '{text}', Confidence: {conf}")




def define_borders(contour):

    # if len(approx) == 4:
    #     return approx.reshape(4, 2)

    hull = cv2.convexHull(contour)
    points = hull.reshape(-1, 2)

    epsilon = 0.02 * cv2.arcLength(points, True)
    approx = cv2.approxPolyDP(points, epsilon, True)
    approx = approx.reshape(-1, 2)
    return approx



def find_largest_contour(image):
    # Read the imag
    if isinstance(image, str):
        image = cv2.imread(image)
    orig = image.copy()
    h, w = image.shape[:2]
    orig = cv2.resize(orig, (w // 2, h //2))

    # Preprocess
    processer = ImageProcessor(
        grayscale=True,
        contrast_clip_limit=10,
        canny_blur_kernel=(5, 5),
        canny_low_threshold=30,
        canny_high_threshold=90,
        dilate_kernel_size=(3,3),
        dilate_iterations=1
    )
    processed = processer.preprocess_image(orig)

    cv2.imshow("Processed", processed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    contours, _ = cv2.findContours(processed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) >= 4:
            valid_contours.append(contour)

    # Sort by area (largest first)
    valid_contours.sort(key=cv2.contourArea, reverse=True)


    return orig, valid_contours[0]


def crop_card(image, points):

    ordered_points = order_points(points)

    (tl, tr, br, bl) = ordered_points

    width_top = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    width_bottom = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    max_width = max(int(width_top), int(width_bottom))

    height_left = np.sqrt(((bl[0] - tl[0]) ** 2) + ((bl[1] - tl[1]) ** 2))
    height_right = np.sqrt(((br[0] - tr[0]) ** 2) + ((br[1] - tr[1]) ** 2))
    max_height = max(int(height_left), int(height_right))



    # Define destination points
    dst_points = np.array([
        [0, 0],                        # top-left
        [max_width - 1, 0],            # top-right
        [max_width - 1, max_height - 1], # bottom-right
        [0, max_height - 1]            # bottom-left
    ], dtype="float32")

    # Get perspective transform matrix
    transform_matrix = cv2.getPerspectiveTransform(ordered_points.astype(dtype="float32"), dst_points)

    # Apply perspective transformation
    cropped = cv2.warpPerspective(image, transform_matrix, (max_width, max_height))

    # detect_orientation(cropped)

    return cropped, (max_width, max_height)

def correct_rotation(img, box):
    pass

def order_points(pts):
    print(pts)

    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect


def least_bbox(image):
    if isinstance(image, str):
        image = cv2.imread(image)

    original = image.copy()
    text_presence, rects = detect_text_presence(original)

    if text_presence:
        x1 = image.shape[1] + 1
        y1 = image.shape[0] + 1
        x2 = 0
        y2 = 0
        for rect in rects:
            x1 = min(x1, rect[0][0])
            y1 = min(y1, rect[0][1])
            x2 = max(x2, rect[1][0])
            y2 = max(y2, rect[1][1])

        scale_factor = 0.01
        H, W = image.shape[:2]
        x1 = max(0, x1 - W * scale_factor)
        y1 = max(0, y1 - H * scale_factor)
        x2 = min(image.shape[1], x2 + W * scale_factor)
        y2 = min(image.shape[0], y2 + H * scale_factor)

        # drawContours(image, np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], dtype=np.int32), size=100)

        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)

    else :
        print("mal9ina walo")
        return []


def get_original_size(bbox, W, H, new_W, new_H):
    rW = W / float(new_W)
    rH = H / float(new_H)

    x1 = int(bbox[0][0] * rW)
    y1 = int(bbox[0][1] * rH)
    x2 = int(bbox[2][0] * rW)
    y2 = int(bbox[2][1] * rH)

    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)


def warp_img(input_path):
    if isinstance(input_path, str):
        input_path = cv2.imread(input_path)

    # background_fixed = fix_borders(input_path)
    # output, contour = find_largest_contour(background_fixed)

    # drawContours(output, contour)
    # drawContours(output, [contour])

    # box = np.array(define_borders(contour))

    processor = ImageProcessor(
            resize="constant:1200",
        )
    output = processor.preprocess_image(input_path)


    box = least_bbox(output)
    print("box : ")
    print(box)
    drawContours(output, [box], size=100)

    cropped, dim = crop_card(output, box)
    if dim[0] < dim[1]:
        cropped = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)

    stretched = cv2.resize(cropped, (int(aspect_ratio * dim[1]), dim[1]), cv2.INTER_CUBIC)

    cv2.imshow("cropped", stretched)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return stretched



if __name__ == '__main__':
    folder="./confidentiels/"
    imgs = [f"output-{i}.png" for i in range(7, 8)]
    for img in imgs:
        warp_img(folder, img)
