import cv2
import numpy as np
from matplotlib import pyplot as plt


def import_image(path):
    img = cv2.imread(path, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    return img, img_gray


def apply_gaussian_blur(img_gray, kernel_size=5, sigma=1.4):
    gaussian_blur = cv2.GaussianBlur(
        img_gray, (kernel_size, kernel_size), sigma, sigma, cv2.BORDER_DEFAULT
    )
    return gaussian_blur


def apply_binary_threshold(img, threshold=128):
    height, width = img.shape
    binary_image = np.copy(img)
    for i in range(height):
        for j in range(width):
            if img[i, j] >= threshold:
                binary_image[i, j] = 255
            else:
                binary_image[i, j] = 0
    return binary_image


def get_sobels(binary_img):
    sobelx = cv2.Sobel(binary_img, cv2.CV_8UC1, 1, 0, ksize=5)
    sobely = cv2.Sobel(binary_img, cv2.CV_8UC1, 0, 1, ksize=5)
    return sobelx, sobely


def get_edges(binary_img, threshold1=50, threshold2=150, apertureSize=3):
    edges = cv2.Canny(binary_img, threshold1, threshold2, apertureSize)
    return edges


def get_points_full_line(l, width, height):
    x1, y1, x2, y2 = l[0], l[1], l[2], l[3]

    if x1 == x2:
        return (x1, 0), (x2, height)

    if y1 == y2:
        return (0, y1), (width, y1)

    slope = (y2 - y1) / (x2 - x1)
    ord_origin = y1 - slope * x1

    if y1 < y2:
        return (0, round(ord_origin)), (round((height - ord_origin) / slope), height)
    elif y1 > y2:
        return (0, round(ord_origin)), (round((-ord_origin) / slope), 0)


def check_line_already_existing(point1, point2, points, epsilon):
    for (x1, y1), (x2, y2) in points:
        if np.sqrt((point1[0] - x1) ** 2 + (point1[1] - y1) ** 2) < epsilon and np.sqrt(
            (point2[0] - x2) ** 2 + (point2[1] - y2) ** 2
        ):
            return True
    return False


def dist(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def get_intersection_between_2_lines(
    line1, line2
):  # line is characterized by two points : line = (p1, p2)
    (x1, y1), (x1prime, y1prime) = line1
    (x2, y2), (x2prime, y2prime) = line2
    if x1 == x1prime:
        if x2 == x2prime:
            return None
        slope2 = (y2 - y2prime) / (x2 - x2prime)
        ord_origin2 = y2 - slope2 * x2
        return (x1, round(slope2 * x1 + ord_origin2))
    elif x2 == x2prime:
        slope1 = (y1 - y1prime) / (x1 - x1prime)
        ord_origin1 = y1 - slope1 * x1
        return (x2, round(slope1 * x2 + ord_origin1))
    else:
        slope1 = (y1 - y1prime) / (x1 - x1prime)
        ord_origin1 = y1 - slope1 * x1
        slope2 = (y2 - y2prime) / (x2 - x2prime)
        ord_origin2 = y2 - slope2 * x2
        if slope1 == slope2:
            return None
        x_int = -(ord_origin2 - ord_origin1) / (slope2 - slope1)
        y_int = slope1 * x_int + ord_origin1
        return (round(x_int), round(y_int))


def get_intersections(points):
    intersections = []
    for i in range(len(points) - 1):
        for j in range(i + 1, len(points)):
            intersections.append(get_intersection_between_2_lines(points[i], points[j]))
    return intersections


def get_houghlines(edges, threshold=28, probabilistic=True):
    if not probabilistic:
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 50, None, 0, 0)
    else:
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold, None, 50, 10)
    return lines


def get_median_length_linesP(linesP):
    length_lines = []
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            length_lines.append(dist((l[0], l[1]), (l[2], l[3])))
    return np.median(length_lines)


def get_points(linesP, median_length_lines, img_shape, use_median=True):
    points = []
    # Draw the lines
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            if use_median:
                if (
                    median_length_lines - 20
                    < dist((l[0], l[1]), (l[2], l[3]))
                    < median_length_lines + 20
                ):
                    point1, point2 = get_points_full_line(l, img_shape[1], img_shape[0])
                    if not check_line_already_existing(point1, point2, points, 30):
                        points.append((point1, point2))

            else:
                point1, point2 = get_points_full_line(l, img_shape[1], img_shape[0])
                if not check_line_already_existing(point1, point2, points, 30):
                    points.append((point1, point2))

    return points


def clean_intersections(intersections, img_shape):
    clean_intersections = []

    for point in intersections:
        if point is None:
            continue
        x, y = point[0], point[1]
        if 0 <= x <= img_shape[1] and 0 <= y <= img_shape[0]:
            clean_intersections.append(point)

    return clean_intersections


def moyenne(points):
    return round(np.mean([point[0] for point in points])), round(
        np.mean([point[1] for point in points])
    )


def clusteriser_intersections(intersections, epsilon=20):
    clusters = {}

    for i, point in enumerate(intersections):
        if point is None:
            continue
        already_in_cluster = False
        for k in clusters.keys():
            if dist(point, moyenne(clusters[k])) <= epsilon:
                already_in_cluster = True
                clusters[k].append(point)
        if not already_in_cluster:
            clusters[i] = [point]

    return clusters


def group_clusters(clusters):
    new_intersections = []

    for k in clusters.keys():
        new_intersections.append(moyenne(clusters[k]))

    return new_intersections


def moyenne_y(points):
    return round(np.mean([point[1] for point in points]))


def moyenne_x(points):
    return round(np.mean([point[0] for point in points]))


def dist_y(y1, y2):
    return np.abs(y1 - y2)


def dist_x(x1, x2):
    return np.abs(x1 - x2)


def cluster_along_y(intersections, epsilon_y):
    clusters_y = {}

    for i, point in enumerate(intersections):
        already_in_cluster = False
        for k in clusters_y.keys():
            if dist_y(point[1], moyenne_y(clusters_y[k])) <= epsilon_y:
                already_in_cluster = True
                clusters_y[k].append(point)
        if not already_in_cluster:
            clusters_y[i] = [point]

    return clusters_y


def cluster_along_x(intersections, epsilon_x):
    clusters_x = {}

    for i, point in enumerate(intersections):
        already_in_cluster = False
        for k in clusters_x.keys():
            if dist_x(point[0], moyenne_x(clusters_x[k])) <= epsilon_x:
                already_in_cluster = True
                clusters_x[k].append(point)
        if not already_in_cluster:
            clusters_x[i] = [point]

    return clusters_x


def remove_y_anomalies(anomaly_points, lines_interests_points_y, clusters_y):
    for k in clusters_y.keys():
        if len(clusters_y[k]) <= 3:
            for point in clusters_y[k]:
                anomaly_points.append(point)
        else:
            lines_interests_points_y[k] = clusters_y[k]
    return anomaly_points, lines_interests_points_y


def remove_x_anomalies(anomaly_points, lines_interests_points_x, clusters_x):
    for k in clusters_x.keys():
        if len(clusters_x[k]) <= 3:
            for point in clusters_x[k]:
                anomaly_points.append(point)
        else:
            lines_interests_points_x[k] = clusters_x[k]
    return anomaly_points, lines_interests_points_x


def remove_anomalies(intersections, epsilon_x=25, epsilon_y=10):

    clusters_x = cluster_along_x(intersections, epsilon_x)
    anomaly_points, lines_interests_points_x = remove_x_anomalies([], {}, clusters_x)

    new_intersections = [
        point for line_x in lines_interests_points_x.values() for point in line_x
    ]
    clusters_y = cluster_along_y(new_intersections, epsilon_y)
    anomaly_points, lines_interests_points_y = remove_y_anomalies(
        anomaly_points, {}, clusters_y
    )

    for k, v in lines_interests_points_y.items():
        lines_interests_points_y[k] = list(set(v))

    return lines_interests_points_y, anomaly_points


def sort_lines_of_interest_by_y(lines_interests_points_y):

    # cols = {}
    cols = []
    moyennes_y = [
        moyenne_y(lines_interests_points_y[k]) for k in lines_interests_points_y.keys()
    ]
    moyennes_y_sorted = sorted(moyennes_y)

    for i, letter in zip(range(len(moyennes_y)), "abcdefghijkl"[: len(moyennes_y)]):
        mean = moyennes_y_sorted[i]
        corresponding_k = None
        for k in lines_interests_points_y.keys():
            if moyenne_y(lines_interests_points_y[k]) == mean:
                corresponding_k = k
        # cols[letter] = lines_interests_points_y[corresponding_k]
        cols.append(lines_interests_points_y[corresponding_k])

    # for k in cols.keys():
    for k in range(len(cols)):
        cols[k].sort(key=lambda x: x[0])

    return cols


def remove_chessboard_contours(cols_sorted):
    if len(cols_sorted) == 9 and len(cols_sorted[0]) == 9:
        return cols_sorted

    if len(cols_sorted) == 11:
        del cols_sorted[0]
        del cols_sorted[-1]

    elif len(cols_sorted) == 10:
        nb = len(cols_sorted)
        means_y = [moyenne_y(cols_sorted[i]) for i in range(nb)]
        mean_y_distance = np.mean([means_y[i + 1] - means_y[i] for i in range(nb - 1)])
        if means_y[1] - means_y[0] < means_y[-1] - means_y[-2]:
            del cols_sorted[0]
        elif means_y[1] - means_y[0] > means_y[-1] - means_y[-2]:
            del cols_sorted[-1]

    for i in range(len(cols_sorted)):
        if len(cols_sorted[i]) == 11:
            del cols_sorted[i][0]
            del cols_sorted[i][-1]

        elif len(cols_sorted[i]) == 10:
            nb = len(cols_sorted[i])
            mean_x_distance = np.mean(
                [cols_sorted[i][j + 1][0] - cols_sorted[i][j][0] for j in range(nb - 1)]
            )
            if cols_sorted[i][1][0] - cols_sorted[i][0][0] < mean_x_distance:
                del cols_sorted[i][0]
            elif cols_sorted[i][-1][0] - cols_sorted[i][-2][0] < mean_x_distance:
                del cols_sorted[i][-1]

    return cols_sorted


def get_chessboard_boxes(cols_sorted):
    boxes = {}
    letters = "abcdefgh"
    nums = "12345678"
    for i in range(8):
        for j in range(8):
            letter = letters[i]
            num = nums[j]
            boxes[letter + num] = [
                cols_sorted[i][j],
                cols_sorted[i][j + 1],
                cols_sorted[i + 1][j],
                cols_sorted[i + 1][j + 1],
            ]

    return boxes


def box_detection(img_path, binary_threshold, use_median, hough_lines_threshold):
    img, img_gray = import_image(img_path)
    gaussian_blur = apply_gaussian_blur(img_gray)

    binary_img = apply_binary_threshold(gaussian_blur, binary_threshold)
    edges = get_edges(binary_img)
    linesP = get_houghlines(edges, hough_lines_threshold)
    median_length_lines = get_median_length_linesP(linesP)

    points = get_points(linesP, median_length_lines, img.shape, use_median)

    intersections = get_intersections(points)
    intersections_cleaned = clean_intersections(intersections, img.shape)

    clusters = clusteriser_intersections(intersections_cleaned)
    new_intersections = group_clusters(clusters)

    lines_interests_points_y, anomaly_points = remove_anomalies(new_intersections)
    cols_sorted = sort_lines_of_interest_by_y(lines_interests_points_y)

    cols_sorted = remove_chessboard_contours(cols_sorted)
    boxes = get_chessboard_boxes(cols_sorted)
    return img, boxes


if __name__ == "__main__":

    img_path = "photos_test/chess.com/4.png"

    binary_threshold = 200
    use_median = False
    hough_lines_threshold = 150

    img, boxes = box_detection(
        img_path, binary_threshold, use_median, hough_lines_threshold
    )

    for point in boxes["e4"]:
        cv2.circle(img, (point[0], point[1]), 5, (0, 0, 255), -1)

    for point in boxes["h3"]:
        cv2.circle(img, (point[0], point[1]), 5, (0, 0, 255), -1)

    for point in boxes["b6"]:
        cv2.circle(img, (point[0], point[1]), 5, (0, 0, 255), -1)

    for point in boxes["f1"]:
        cv2.circle(img, (point[0], point[1]), 5, (0, 0, 255), -1)

    plt.subplots(1, 1, figsize=(10, 7))

    plt.subplot(1, 1, 1), plt.imshow(img)
    plt.title("Highlighted Boxes"), plt.xticks([]), plt.yticks([])
    plt.show()
