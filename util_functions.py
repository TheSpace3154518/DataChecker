import cv2
from Preprocessing import ImageProcessor
import time

LastTime = time.time()
def calculateTime():
    global LastTime
    currentTime = time.time()
    elapsedTime = currentTime - LastTime
    LastTime = currentTime
    return elapsedTime


def drawContours(orig, contours, size=100):
    img = orig.copy()
    cv2.drawContours(img, contours, -1, (0,0, 255), 3)
    processor = ImageProcessor(
            resize=size
        )
    resized = processor.preprocess_image(img)
    cv2.imshow("Edge detection", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return resized
