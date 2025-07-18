import cv2
import numpy as np


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

    new_H = (H // 32) * 32 if H % 32 < 16 else ((H // 32) + 1) * 32
    new_W = (W // 32) * 32 if W % 32 < 16 else ((W // 32) + 1) * 32
    rW = W / float(new_W)
    rH = H / float(new_H)

    # Resize image
    resized = cv2.resize(image, (new_W, new_H))

    # Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create blob from image
    blob = cv2.dnn.blobFromImage(image, 1.0, (new_W, new_H),
                                (123.68, 116.78, 103.94), swapRB=True, crop=False)

    # Set input and run forward pass
    net.setInput(blob)
    (scores, geometry) = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])

    # Decode the predictions
    rectangles = []
    confidences = []

    for y in range(0, scores.shape[2]):
        scores_data = scores[0, 0, y]
        x_data0 = geometry[0, 0, y]
        x_data1 = geometry[0, 1, y]
        x_data2 = geometry[0, 2, y]
        x_data3 = geometry[0, 3, y]
        angles_data = geometry[0, 4, y]

        for x in range(0, scores.shape[3]):
            if scores_data[x] < confidence_threshold:
                continue

            # Calculate offset
            offset_x = x * 4.0
            offset_y = y * 4.0

            # Extract rotation angle
            angle = angles_data[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # Calculate dimensions
            h = x_data0[x] + x_data2[x]
            w = x_data1[x] + x_data3[x]

            # Calculate rotated rectangle
            end_x = int(offset_x + (cos * x_data1[x]) + (sin * x_data2[x]))
            end_y = int(offset_y - (sin * x_data1[x]) + (cos * x_data2[x]))
            start_x = int(end_x - w)
            start_y = int(end_y - h)

            rectangles.append((start_x, start_y, end_x, end_y))
            confidences.append(scores_data[x])

    # Apply non-maximum suppression
    boxes = cv2.dnn.NMSBoxes(rectangles, confidences, confidence_threshold, 0.4)

    # Draw results
    result_image = original.copy()
    text_detected = False

    all_rects = []

    if len(boxes) > 0:
        text_detected = True
        for i in boxes:
            # Scale coordinates back to original image
            x1 = int(rectangles[i][0] * rW)
            y1 = int(rectangles[i][1] * rH)
            x2 = int(rectangles[i][2] * rW)
            y2 = int(rectangles[i][3] * rH)

            cv2.rectangle(result_image, [x1, y1], [x2, y2], (0, 0, 255), 1)
            all_rects.append([[x1, y1],[x2, y2]])

    print("Detected !!")
    cv2.imshow("BBOX", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return text_detected, np.array(all_rects, dtype=np.int32)


if __name__ == "__main__":
    # Example usage
    image_path = "./Recruits/cin-1.png"
    detect_text_presence2(image_path)

    # text_detected, result_image = detect_text_presence(image_path)

    # Display result
    cv2.imshow("Text Detection", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
