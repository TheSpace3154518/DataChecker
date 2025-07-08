import cv2
import numpy as np


def drawContours(orig, contours):
    img = orig.copy()
    cv2.drawContours(img, contours, -1, (0,0, 255), 3)
    cv2.imshow("Edge detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def define_borders(contour):

    # if len(approx) == 4:
    #     return approx.reshape(4, 2)

    hull = cv2.convexHull(contour)
    points = hull.reshape(-1, 2)

    epsilon = 0.02 * cv2.arcLength(points, True)
    approx = cv2.approxPolyDP(points, epsilon, True)
    approx = approx.reshape(-1, 2)
    return approx



def find_largest_contour(image_path):
    # Step 1: Read the image
    image = cv2.imread(image_path)
    orig = image.copy()
    h, w = image.shape[:2]
    orig = cv2.resize(orig, (w // 2, h //2))

    # Step 2: Preprocess
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 150)

    kernel = np.ones((3, 3))
    edged = cv2.dilate(edged, kernel, iterations=1)

    cv2.imshow("Edge", edged)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Step 3: Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

    # Define destination points (rectangle)
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

    return cropped, (max_width, max_height)

def correct_rotation(img, box):
    pass

def order_points(pts):

    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect



# Example usage
if __name__ == "__main__":
    input_path = "./Recruits/warped.jpeg"
    output, contour = find_largest_contour(input_path)

    drawContours(output, [contour])

    box = define_borders(contour)

    drawContours(output, [box])


    print(box)
    cropped, dim = crop_card(output,np.array(box))
    if dim[0] < dim[1]:
        cropped = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)

    cv2.imshow("cropped", cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
