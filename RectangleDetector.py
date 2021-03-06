import cv2 as cv
import numpy as np
import copy


class RectangleDetector(object):
    """
        Wololooo
    """

    def __init__(self, image, denoising_h=10, canny_aperture_size=5,
                 canny_l2gradient=True, canny_thr1=20, canny_thr2=100):
        if image is None:
            return
        self.denoising_h = denoising_h
        self.canny_aperture_size = canny_aperture_size
        self.canny_l2gradient = canny_l2gradient
        self.canny_thr1 = canny_thr1
        self.canny_thr2 = canny_thr2
        self.org_img = image
        self.org_img = cv.resize(image, None, fx=0.2, fy=0.2, interpolation=cv.INTER_CUBIC)
        self.canny_img = None
        self.contours = None
        self.card_contour_list = None
        self.card_corners_list = None

    def crop(self):
        print("preprocessing")
        self.canny_img = self.preprocess_image()
        print("finding contours")
        self.contours = self.find_sort_contours()
        print("validating contours")
        self.validate_contours()
        return self.crop_cards()

    def crop_cards(self):
        cropped_images = []
        for corners in self.card_corners_list:
            cropped_image = self.crop_card(corners)
            cropped_images.append(cropped_image)
        return cropped_images

    def crop_card(self, corners):
        dst = np.float32([[0, 0], [630, 0], [630, 880], [0, 880]])
        src = np.float32(corners)
        mat = cv.getPerspectiveTransform(src, dst)
        return cv.warpPerspective(self.org_img, mat, (630, 880))

    def preprocess_image(self):
        img = cv.cvtColor(self.org_img, cv.COLOR_BGR2GRAY)
        denoised = cv.bilateralFilter(img, 9, 75, 75)
        v = np.median(denoised)
        # apply automatic Canny edge detection using the computed median
        self.canny_thr1 = int(max(0, (1.0 - 0.33) * v))
        self.canny_thr2 = int(min(255, (1.0 + 0.33) * v))

        canny = cv.Canny(denoised, self.canny_thr1, self.canny_thr2,
                         apertureSize=self.canny_aperture_size, L2gradient=self.canny_l2gradient)
        return canny

    def find_sort_contours(self):
        contours, _ = cv.findContours(self.canny_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        return sorted(contours, key=cv.contourArea, reverse=True)[0:5]

    def validate_contours(self):
        self.card_corners_list = []
        self.card_contour_list = []
        for contour in self.contours:
            corners = self.get_corners(contour)
            if len(corners) == 4:
                lu, ru, rb, lb, width, height = self.sort_corners_calculate_width_height(corners)
                if self.validate_aspect_ratio(width, height):
                    self.card_corners_list.append([lu, ru, rb, lb])
                    self.card_contour_list.append(contour)

    @staticmethod
    def validate_aspect_ratio(width, height, radius=0.3):
        # TODO: Calculate two widths and two heights .. maybe could help with the big radius
        ratio = height / width
        if (ratio > (1.3968325 - radius)) and (ratio < (1.3968325 + radius)):
            return True
        else:
            return False

    @staticmethod
    def sort_corners_calculate_width_height(corners):
        sorted_corners = []
        for c in corners:
            sorted_corners.append(c[0])
        sorted_corners = np.array(sorted_corners)
        corners = sorted_corners
        s = corners.sum(axis=1)
        lu = corners[np.argmin(s)]
        rb = corners[np.argmax(s)]

        diff = np.diff(corners, axis=1)
        ru = corners[np.argmin(diff)]
        lb = corners[np.argmax(diff)]

        distances = [RectangleDetector.calc_eucl_dist(lu, ru), RectangleDetector.calc_eucl_dist(lu, rb),
                     RectangleDetector.calc_eucl_dist(lu, lb)]
        index = distances.index(min(distances))
        if index == 0:
            # closest to left upper is right upper
            width = distances[0]
            height = distances[2]
            return lu, ru, rb, lb, width, height
        if index == 1:
            # closest to left upper is right bottom
            # error?
            pass
        if index == 2:
            # closest to left upper is left bottom
            width = distances[2]
            height = distances[0]
            return lb, lu, ru, rb, width, height


    @staticmethod
    def calc_eucl_dist(p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    @staticmethod
    def get_corners(contour):
        arc_length = cv.arcLength(contour, True)
        return cv.approxPolyDP(contour, 0.04 * arc_length, True)

    @staticmethod
    def draw_contours(img, contours, colour):
        img_copy = copy.deepcopy(img)
        img_copy = cv.drawContours(img_copy, contours, -1, colour, 2)
        return img_copy

    @staticmethod
    def draw_corners(img, corners, colour):
        for _corners in corners:
            for corner in _corners:
                cv.circle(img, tuple(corner), 3, colour, -1)

#

# TODO: Do not write file if it is not cropped.
#   it does, but it is warped (wrong corners)
#   the problem is that the card i more than 95% of the image size ...
# TODO: It does not work on already cropped images, why ?
