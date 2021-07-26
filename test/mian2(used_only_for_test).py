import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from numpy.matlib import repmat
import matplotlib.patches as mpatches
from skimage.measure import label, regionprops

def normalize(x):
    """
    :param x: the image matrix¸
    :return:  resized image matrix
    """

    Max = np.max(x.flatten())
    Min = np.min(x.flatten())
    x = (x - Min) / (Max - Min)
    return x


def kmeans(data, n_cl, verbose=False):
    n_samples, dim = data.shape

    # init the center point
    centers = data[np.random.choice(range(n_samples), size=n_cl)]
    old_labels = np.zeros(shape=n_samples)

    while True:

        # sharing the sample
        distances = np.zeros(shape=(n_samples, n_cl))
        for c_idx, c in enumerate(centers):
            distances[:, c_idx] = np.sum(np.square(data - repmat(c, n_samples, 1)), axis=1)

        new_labels = np.argmin(distances, axis=1)

        # estimate the sample
        for l in range(0, n_cl):
            centers[l] = np.mean(data[new_labels == l], axis=0)

        if verbose:
            fig, ax = plt.subplots()
            ax.scatter(data[:, 0], data[:, 1], c=new_labels, s=40)
            ax.plot(centers[:, 0], centers[:, 1], 'r*', markersize=20)
            plt.waitforbuttonpress()
            plt.close()

        if np.all(new_labels == old_labels):
            break

        # update the iterations
        old_labels = new_labels

    return new_labels


def main2(imagepath):
    img = cv2.imread(imagepath)
    # convert into grayscale

    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # apply Gaussian Blur to the image to reduce noise

    im_gau = cv2.GaussianBlur(im_gray, (3, 3), 0)

    # apply Canny edge deception
    canny = cv2.Canny(img, 150, 250)

    ret, binary_img = cv2.threshold(canny, 127, 255, cv2.THRESH_BINARY)

    # Morphological operations
    kernel = np.ones((15, 25), np.uint8)

    # Close the operation first to connect the digital part of the license plate,
    # and then open the operation to remove the block or smaller parts
    close_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
    open_img = cv2.morphologyEx(close_img, cv2.MORPH_OPEN, kernel)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilation_img = cv2.dilate(open_img, element, iterations=3)

    # Obtain the borders
    contours, hierarchy = cv2.findContours(dilation_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = []
    for k in range(len(contours)):
        area.append(cv2.contourArea(contours[k]))
        max_idx = np.argmax(np.array(area))
    rectangles = []

    x = []
    y = []
    for point in contours[max_idx]:
        y.append(point[0][0])
        x.append(point[0][1])
    r = [min(y), min(x), max(y), max(x)]
    rectangles.append(r)

    # Identify the license plate area by color
    # When setting the color recognition lower limit low,
    # it can be adjusted according to the recognition result
    dist_r = []
    max_mean = 0
    for r in rectangles:
        block = img[r[1]:r[3], r[0]:r[2]]
        hsv = cv2.cvtColor(block, cv2.COLOR_BGR2HSV)
        low = np.array([100, 60, 60])
        up = np.array([255, 255, 255])
        result = cv2.inRange(hsv, low, up)

        # Calculate the average to find the areas with most blue stuff
        mean = cv2.mean(result)
        if mean[0] > max_mean:
            max_mean = mean[0]
            dist_r = r

    # Crop location map
    cropped = img[dist_r[1] + 45:dist_r[3] - 15, dist_r[0] + 5:dist_r[2] - 5]  # 裁剪坐标为[y0:y1, x0:x1]
    cv2.imwrite("./cv_cut22.jpg", cropped)

    n_cl = 2
    img = cropped
    # Load the image
    h, w, c = img.shape
    # Add coordinates
    row_indexes = np.arange(0, h)
    col_indexes = np.arange(0, w)
    coordinates = np.zeros(shape=(h, w, 2))
    coordinates[..., 0] = normalize(repmat(row_indexes, w, 1).T)
    coordinates[..., 1] = normalize(repmat(col_indexes, h, 1))

    data = np.concatenate((img, coordinates), axis=-1)
    data = np.reshape(data, newshape=(w * h, 5))

    # using main function of K-Means
    labels = kmeans(data, n_cl=n_cl, verbose=False)
    result = np.reshape(labels, (h, w))
    result = result.astype('uint8')
    ret, result = cv2.threshold(result * 255, 127, 255, cv2.THRESH_BINARY)

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))  # Morphological denoising
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, element)  # Open operation denoising
    plt.imshow(result, cmap='gray')
    plt.show()

    label_image, num = label(result, connectivity=1, background=0, return_num=True)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 6))
    ax0.imshow(result, plt.cm.gray)
    ax1.imshow(img)
    z = 0
    for region in regionprops(label_image):  # Loop to get the attribute set of each connected region
        z += 1
        # Ignore small areas
        if region.area < 100 or region.area > 500:
            continue
        # Draw outsourcing rectangle
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
        patch = img[minr:maxr, minc:maxc]  # 裁剪坐标为[y0:y1, x0:x1]

        try:
            for i in range(15):
                path_name = 'path'
                del_name = path_name + str(i) + ".jpg"
                os.remove(del_name)
        except:
            pass

        path_name = 'path' + str(z) + ".jpg"
        cv2.imwrite(path_name, "test/")
        ax1.add_patch(rect)
    fig.tight_layout()
    plt.show()
