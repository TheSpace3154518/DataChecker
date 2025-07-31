import cv2
import math
import pandas as pd
from util_functions import drawContours
from collections import Counter
import numpy as np
from scipy import stats


def rotate_image(img, angle):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    height, width = img.shape[:2]
    center = (width // 2, height // 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)

    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])

    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))

    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]

    rotated_img = cv2.warpAffine(img, rotation_matrix, (new_width, new_height))

    # rotated_img = cv2.cvtColor(rotated_img, cv2.COLOR_RGB2BGR)
    return rotated_img


def adjust_gamma(image, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)


def detect_text_presence(image_path, model_path="./models/frozen_east_text_detection.pb", confidence_threshold=0.5):
    print("detecting...")
    try:
        # Load EAST model
        net = cv2.dnn.readNet(model_path)
    except:
        return False, None

    # Load image
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
    else :
        image = image_path

    original = image.copy()
    (H, W) = image.shape[:2]

    new_H = (H // 32) * 32
    new_W = (W // 32) * 32
    rW = W / float(new_W)
    rH = H / float(new_H)

    # Resize image
    resized = cv2.resize(image, (new_W, new_H))

    # Convert to RGB
    # resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Create blob from image
    blob = cv2.dnn.blobFromImage(image, 1.0, (new_W, new_H),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)

    # Set input and run forward pass
    net.setInput(blob)
    (scores, geometry) = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])

    height, width = scores.shape[2:4]

    rectangles = []
    confidences = []
    angles = []


    for y in range(height):
        scores_data = scores[0, 0, y]
        x0_data = geometry[0, 0, y]
        x1_data = geometry[0, 1, y]
        x2_data = geometry[0, 2, y]
        x3_data = geometry[0, 3, y]
        angles_data = geometry[0, 4, y]

        for x in range(width):
            score = scores_data[x]
            if score < confidence_threshold:
                continue

            # Offsets
            offset_x = x * 4.0
            offset_y = y * 4.0

            # Geometry
            angle = angles_data[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]

            if h > w :
                angle -= math.pi/2
                h, w = w, h

            # Compute box center
            end_x = int(offset_x + (cos * x1_data[x]) + (sin * x2_data[x]))
            end_y = int(offset_y - (sin * x1_data[x]) + (cos * x2_data[x]))
            start_x = int(end_x - w)
            start_y = int(end_y - h)


            angles.append(int(-angle * 180.0 / math.pi))
            rectangles.append([start_x, start_y, int(w), int(h)])
            confidences.append(float(score))


    if len(angles) > 0:
        start = 0
        mode = stats.mode(angles[start:])
        mode_angle = mode.mode
        print(f"{mode_angle} | {mode.count/(len(angles) - start) : .2f}")

        correction_factor = 1.05
        dominant_angle = round(mode_angle * correction_factor)
        print(dominant_angle)


        threshold = 3
        if abs(dominant_angle) >= threshold :
            # image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
            original = rotate_image(image, dominant_angle)
            cv2.imshow("Rotated", original)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            return detect_text_presence(original)


    boxes = np.array(rectangles)

    nms_boxes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)


    # Draw results
    result_image = original.copy()
    text_detected = False

    all_rects = []




    if len(nms_boxes) > 0:
        text_detected = True
        for i in nms_boxes:
            # Scale coordinates back to original image
            x1 = int(boxes[i][0] * rW)
            y1 = int(boxes[i][1] * rH)
            x2 = int((boxes[i][0] + boxes[i][2]) * rW)
            y2 = int((boxes[i][1] + boxes[i][3]) * rH)
            cv2.rectangle(result_image, [x1, y1], [x2, y2], (0, 0, 255), 1)
            all_rects.append([[x1, y1],[x2, y2]])

    print("Detected !!")
    cv2.imshow("BBOX", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    all_rects = np.array(all_rects, dtype=np.int32)
    return original, text_detected, all_rects


if __name__ == "__main__":
    # Example usage
    image_path = "./Recruits/cin-1.png"

    # text_detected, result_image = detect_text_presence(image_path)

    # Display result
    cv2.imshow("Text Detection", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
