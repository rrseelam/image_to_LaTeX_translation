"""
Data Preprocessing Script

- enumerates data in the raw_datasets/ and places stardized ouput pickles in the processes_dataset folder
- converts to standard format (jpeg)
- drops unsupported characters
- converts to custom bool encoding format
"""

import numpy as np
import cv2 as cv

from bounding_box_detector import *

from torch.utils.data import Dataset

def main():

    # "raw_datasets/CROHME_test_2011-converted/TestData1_2_sub_29.png"
    fs = ["raw_datasets/CROHME_test_2011-converted/TestData2_2_sub_14.png",
          "raw_datasets/CROHME_test_2011-converted/TestData2_1_sub_2.png",
          "raw_datasets/CROHME_test_2011-converted/TestData2_3_sub_8.png"]

    for f in fs:
        img = chunk_image_path(f, show=True)
        og = cv.imread(f) 
        boxes = grab_bounding_boxes(img)
        print(boxes)
        for box in boxes:
            print(box)
        
            pt1 = (box[2] - 1, box[0] - 1)
            pt2 = (box[3] + 1, box[0] - 1)
            pt3 = (box[3] + 1, box[1] + 1)
            pt4 = (box[2] - 1, box[1] + 1)

            cv.rectangle(og, pt1, pt3, (256, 128, 0), 1)  # Green color, thickness 2
        
        cv.imshow("img", og)
        cv.waitKey(0)
        cv.destroyAllWindows()

    print("This is a ulitiy module")

def chunk_image_path(img_path, show=False):
    # returns an np array of a a image compressed with out compression format
    # expects an image in the form
    
    image = cv.imread(img_path) 

    return chunk_image(image, show)

def filter_image(img, up, grayscale=False):
    # filters image intensity based on lower bound (lb) and upper bound(ub)
    # all values in img < lb get set to 0
    # all values in img >= up get set to 255
    # return a copy of the filtered image

    # Convert image to grayscale
    gray = img.copy()
    if grayscale: 
        gray = cv.cvtColor(gray, cv.COLOR_BGR2GRAY)
        # print(gray.shape)

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

    gray_image = filter_image(gray_image, 200, grayscale=False)

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

class SymbolDataset(Dataset):
    def __init__(self, images, labels, transforms=None):
        self.images = images
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transforms:
            image = self.transforms(image)
        return image, label


if __name__ == "__main__":
    main()