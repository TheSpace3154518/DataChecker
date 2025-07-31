import cv2
import numpy as np

class ImageProcessor:
    def __init__(self,
            grayscale=False,
            resize=100,
            contrast_clip_limit=0,
            contrast_tile_size=(8, 8),
            denoise_h=0,
            sharpness_alpha=1,
            sharpness_beta=0,
            sharpness_sigma=3,
            canny_low_threshold=0,
            canny_high_threshold=255,
            canny_blur_kernel=(5, 5),
            dilate_kernel_size=(3, 3),
            dilate_iterations=0):

        self.grayscale = grayscale
        self.resize = resize
        self.contrast_clip_limit = contrast_clip_limit
        self.contrast_tile_size = contrast_tile_size
        self.denoise_h = denoise_h
        self.sharpness_alpha = sharpness_alpha
        self.sharpness_beta = sharpness_beta
        self.sharpness_sigma = sharpness_sigma
        self.canny_low_threshold = canny_low_threshold
        self.canny_high_threshold = canny_high_threshold
        self.canny_blur_kernel = canny_blur_kernel
        self.dilate_kernel_size = dilate_kernel_size
        self.dilate_iterations = dilate_iterations

    def preprocess_image(self, image):
        # Load the image
        if isinstance(image, str):
            processed = cv2.imread(image)
        else:
            processed = image.copy()

        # Resize
        processed = self.resize_img(processed)


        # Grayscale
        if self.grayscale:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

        # Contrast enhancement
        processed = self.contrast(processed)

        # Denoise
        processed = self.denoise(processed)

        # Sharpness
        processed = self.sharpness(processed)

        # Canny Edge Detection with dilation
        processed = self.edge_detection(processed)

        return processed

    def img_show(self, image):
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def resize_img(self, image):
        if isinstance(self.resize, str) and 'constant' in self.resize:
            target = int(self.resize.split(":")[1])
            scale_percent = (target / image.shape[1]) * 100
        else :
            scale_percent = self.resize
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)

    def contrast(self, image):
        if self.contrast_clip_limit <= 0:
            return image

        clahe = cv2.createCLAHE(clipLimit=self.contrast_clip_limit,
                            tileGridSize=self.contrast_tile_size)

        if len(image.shape) == 3:
            l, a, b = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))

            clahe_l = clahe.apply(l)

            enhanced = cv2.merge([clahe_l, a ,b])

            # Convert back to BGR
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            return enhanced
        else :
            return clahe.apply(image)

    def denoise(self, image):
        if self.denoise_h <= 0:
            return image

        return cv2.fastNlMeansDenoising(image, h=self.denoise_h)

    def sharpness(self, image):
        if self.sharpness_alpha == 1 and self.sharpness_beta == 0:
            return image

        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=self.sharpness_sigma)
        return cv2.addWeighted(image, self.sharpness_alpha, blurred, -self.sharpness_beta, 0)

    def edge_detection(self, image):
        if self.canny_low_threshold <= 0:
            return image

        blurred = cv2.GaussianBlur(image, self.canny_blur_kernel, 0)
        edges = cv2.Canny(blurred, self.canny_low_threshold, self.canny_high_threshold)

        # Apply dilation if iterations > 0
        if self.dilate_iterations > 0:
            kernel = np.ones(self.dilate_kernel_size)
            edges = cv2.dilate(edges, kernel, iterations=self.dilate_iterations)

        return edges



if __name__ == "__main__":
    processor = ImageProcessor()
    image = cv2.imread("./Recruits/cin-1.png")
    cv2.imshow("Processed", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    processed = processor.preprocess_image(image)
    cv2.imshow("Processed", processed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
