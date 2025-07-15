import cv2
import numpy as np
import math
from Preprocessing import preprocess_image

# def decode_predictions(scores, geometry, scoreThresh):
#     detections = []
#     confidences = []

#     height, width = scores.shape[2:4]

#     for y in range(0, height):
#         scoresData = scores[0, 0, y]
#         x0_data = geometry[0, 0, y]
#         x1_data = geometry[0, 1, y]
#         x2_data = geometry[0, 2, y]
#         x3_data = geometry[0, 3, y]
#         anglesData = geometry[0, 4, y]
#         for x in range(0, width):
#             if scoresData[x] < scoreThresh:
#                 continue

#             offsetX, offsetY = x * 4.0, y * 4.0
#             angle = anglesData[x]
#             cos, sin = np.cos(angle), np.sin(angle)
#             h = x0_data[x] + x2_data[x]
#             w = x1_data[x] + x3_data[x]
#             endX = int(offsetX + cos * x1_data[x] + sin * x2_data[x])
#             endY = int(offsetY - sin * x1_data[x] + cos * x2_data[x])
#             startX = int(endX - w)
#             startY = int(endY - h)

#             detections.append(((startX, startY, endX, endY), angle))
#             confidences.append(float(scoresData[x]))

#     return detections, confidences

# # Load image
# image = cv2.imread('./Recruits/cin2.png')
# (H, W) = image.shape[:2]
# orig = image.copy()

# (H, W) = image.shape[:2]
# # orig = cv2.resize(orig, (W // 2 , H // 2))

# # Resize to multiple of 32
# newW = (W // 32) * 32
# newH = (H // 32) * 32
# rW, rH = W / float(newW), H / float(newH)
# image = cv2.resize(image, (newW, newH))

# # Load the EAST model
# net = cv2.dnn.readNet("./models/frozen_east_text_detection.pb")

# # Create a blob and forward pass
# blob = cv2.dnn.blobFromImage(image, 1.0, (newW, newH),
#                              (123.68, 116.78, 103.94), swapRB=True, crop=False)
# net.setInput(blob)
# (scores, geometry) = net.forward(["feature_fusion/Conv_7/Sigmoid",
#                                   "feature_fusion/concat_3"])

# # Decode
# boxes, confidences = decode_predictions(scores, geometry, 0.5)

# # Draw detections with rotation info
# for (startX, startY, endX, endY), angle in boxes:
#     # Scale box back to original size
#     startX = int(startX * rW)
#     startY = int(startY * rH)
#     endX = int(endX * rW)
#     endY = int(endY * rH)

#     # Estimate rotation angle in degrees
#     angle_deg = round(math.degrees(angle), 2)
#     cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
#     cv2.putText(orig, f"{angle_deg}°", (startX, startY - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# # Show result
# cv2.imshow("Text Detection with Orientation", orig)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def draw_bbox(image, bbox):
    cv2.rectangle(image, bbox[0], bbox[1], (0, 0, 255), 1)
    cv2.imshow("output", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def proportion(r1, r2, r3, r4, W, H):
    x1, y1 = W * r1, H * r2
    x2, y2 = x1 + (W - x1) * r3, y1 + (H - y1) * r4
    return x1, y1, x2, y2


def check_model(image):

    # Check Zone
    x1, y1, x2, y2 = proportion(0, 0, 3/8, 1/5, W, H)
    draw_bbox(image, np.array([[x1,y1],[x2,y2]], dtype=np.int32))



if __name__ == "__main__":

    image = cv2.imread('./Recruits/aspct.png')
    H, W = image.shape[:2]


    # Model 1 (l9dim)
    # CIN Zone
    x1, y1, x2, y2 = proportion(2/3, 3/4, 1/2, 1/2, W, H)
    draw_bbox(image, np.array([[x1,y1],[x2,y2]], dtype=np.int32))

    # Check Zone
    x1, y1, x2, y2 = proportion(0, 0, 3/8, 1/5, W, H)
    draw_bbox(image, np.array([[x1,y1],[x2,y2]], dtype=np.int32))

    # Name Zone
    x1, y1, x2, y2 = proportion(0, 1/4, 3/8, 5/12, W, H)
    draw_bbox(image, np.array([[x1,y1],[x2,y2]], dtype=np.int32))



    # Model 2 (jdid)
    image = cv2.imread('./Recruits/cin2.png')
    H, W = image.shape[:2]
    # CIN Zone
    x1, y1, x2, y2 = proportion(9/95, 6/7, 3/14, 1, W, H)
    draw_bbox(image, np.array([[x1,y1],[x2,y2]], dtype=np.int32))

    # Check Zone
    x1, y1, x2, y2 = proportion(0, 0, 3/8, 1/5, W, H)
    draw_bbox(image, np.array([[x1,y1],[x2,y2]], dtype=np.int32))

    # Name Zone
    x1, y1, x2, y2 = proportion(9/25, 1/5, 1/3, 1/3, W, H)
    draw_bbox(image, np.array([[x1,y1],[x2,y2]], dtype=np.int32))
