import cv2 as cv2
import pytesseract as pytesseract
import numpy as np
from PIL import Image
import os


def final_image_operations(image_patch):
    """
    :param image_patch: the path name of the image
    :return: the final processed image for txt recreation
    """

    # read and covert into gray scale
    pic = cv2.imread(image_patch)
    pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)

    # Gaussian smooth
    kernel = np.ones((1, 1), np.uint8)
    pic = cv2.dilate(pic, kernel, iterations=20)
    pic = cv2.erode(pic, kernel, iterations=20)

    pic = cv2.adaptiveThreshold(pic, 300, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

    newPath = image_patch[:-4]
    cv2.imwrite(newPath + "_new.png", pic)

    return newPath + "_new.png"


def covert_to_text(image_patch):
    """
    :param image_patch: the path name of the image
    :return: string representation of the image
    """
    return pytesseract.image_to_string(Image.open(image_patch), lang='eng',
                                       config='--psm 6')


def return_the_text(path):
    # this funcion is created to remove the image that was processed by using final_image_operations
    returnPath = final_image_operations(path)
    text = covert_to_text(returnPath)
    try:
        os.remove(returnPath)
    except:
        pass
    return text


if __name__ == "__main__":

    final_txt = ""

    # load the image and pass it to ORC recognition
    try:
        final_txt = return_the_text("../path1.png")
        if len(final_txt) == 0 or final_txt == "":
            final_txt = "Could not load the text on the image"
    except:
        pass

    # string operations
    text_list = []
    check_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                  'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                  'U', 'V', 'W', 'X', 'Y', 'Z', '1', '2', '3', '4',
                  '5', '6', '7', '8', '9', '0']

    for i in range(len(final_txt)):
        if final_txt[i] not in check_list:
            pass
        else:
            text_list.append(final_txt[i])

    if len(text_list) != 6:
        print("The program may fail or the car plates is costumed")
        print("Please check the car plate manually")

    for elements in text_list:
        print(elements, end="")
