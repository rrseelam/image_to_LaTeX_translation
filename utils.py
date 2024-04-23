"""
Data Preprocessing Script

- enumerates data in the raw_datasets/ and places stardized ouput pickles in the processes_dataset folder
- converts to standard format (jpeg)
- drops unsupported characters
- converts to custom bool encoding format
"""

import numpy as np
import cv2 as cv

from data_preprocessing import *
from bounding_box_detector import *


def main():

    f = "image.png"

    img = chunk_image_path(f, show=True)
    og = cv.imread(f) 
    boxes = grab_bounding_boxes(img)
    print(boxes)
    for box in boxes:
        print(box)
       
        pt1 = (box[2] - 5, box[0] - 5)
        pt2 = (box[3] + 5, box[0] - 5)
        pt3 = (box[3] + 5, box[1] + 5)
        pt4 = (box[2] - 5, box[1] + 5)

        cv.rectangle(og, pt1, pt3, (0, 255, 0), 2)  # Green color, thickness 2
    
    cv.imshow("img", og)
    cv.waitKey(0)
    cv.destroyAllWindows()

    print("This is a ulitiy module")

def chunk_image_path(img_path, show=False):
    # returns an np array of a a image compressed with out compression format
    # expects an image in the form
    
    image = cv.imread(img_path) 

    return chunk_image(image, show)

def filter_image(img, lb, up, grayscale=False):
    # filters image intensity based on lower bound (lb) and upper bound(ub)
    # all values in img < lb get set to 0
    # all values in img >= up get set to 255
    # return a copy of the filtered image

    # Convert image to grayscale
    gray = img.copy()
    if grayscale: 
        gray = cv.cvtColor(gray, cv.COLOR_BGR2GRAY)
        # print(gray.shape)

    gray[gray < lb] = 0
    gray[gray >= up] = 255

    return gray


def chunk_image(img, show=False):
    # returns an np array of a a image compressed with out compression format
    # expects an np array of size H,W,3

    # Load the image

    # Convert the image to grayscale

    if show:
        cv.imshow("img", img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # print(gray_image.shape)

    if show:
        cv.imshow("img", gray_image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    gray_image = filter_image(gray_image, 200, 200, False)

    if show:
        cv.imshow("img", gray_image)
        cv.waitKey(0)
        cv.destroyAllWindows()
   
    return gray_image


def encode_labels(labels):
    # input:
    #   labels -> (np.array) labels of a labeled dataset
    # output:
    #   tuple(class_names, one_hot)
    #    - class_names -> (np.array) unique class names
    #    - one_hot -> (np.array) one-hot encoding of the original labels

    class_names = np.unique(labels)
    one_hot = np.array([np.where(class_names == l)[0][0] for l in labels])
    return class_names, one_hot



if __name__ == "__main__":
    main()