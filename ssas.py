from cv2 import cv2
import numpy as np

image = cv2.imread("Image.jpg")
video = cv2.VideoCapture("Lane Detection Test Video 01.mp4")


def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * 0.87)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    coordinates_array = np.array([x1, y1, x2, y2])
    return coordinates_array


def average_slope_intercept(image, lines):
    left_fit = []  # coordinates for the left overlay line
    right_fit = []  # same thing for the right line
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)  # transforming each line in an array
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    line_array = np.array([left_line, right_line])

    return line_array


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image


def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
        [(350, height), (1250, height), (850, 750)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def line_detection(canny, image):
    lane_image = np.copy(image)
    canny = canny(lane_image)
    cropped_image = region_of_interest(canny)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 50, np.array([]), minLineLength=40, maxLineGap=5)
    average_lines = average_slope_intercept(lane_image, lines)
    line_image = display_lines(lane_image, average_lines)
    combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
    cv2.imshow("Image", combo_image)
    cv2.waitKey(0)


def videocapture():
    while video.isOpened():
        ret, frame = video.read()
        canny = canny(frame)
        cropped_image = region_of_interest(canny)
        lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 50, np.array([]), minLineLength=40, maxLineGap=5)
        average_lines = average_slope_intercept(frame, lines)
        line_image = display_lines(frame, average_lines)
        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        cv2.imshow("Image", combo_image)
        cv2.waitKey(1)

        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

    video.release()
    cv2.destroyAllWindows()


videocapture()
#line_detection(canny, image2)
# print(image.shape)
