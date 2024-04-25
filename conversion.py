import torch
import cv2
import utils as ut


def main():
    print("Testing Code")
    pass

class_names = ['(' ')' '+' '-' '0' '1' '2' '3' '4' '5' '6' '7' '8' '9' '=' 'C' 'a' 'b'
 'd' 'dot' 'e' 'f' 'g' 'h' 'i' 'j' 'k' 'l' 'm' 'n' 'p' 'pi' 'q' 'r' 's'
 'slash' 't' 'u' 'v' 'w' 'x' 'y' 'z']

def predict(img, model):
    img = img.tensor()
    img = img.repeat(1, 3, 1, 1)
    outputs = model(img)
    _, preds = torch.max(outputs, 1)


    # given an img (32 x 32) return predicted class
    return class_names[preds[0]]


size = (32, 32)
def convert_numpy_array(img):
    
    processed = img
    processed = cv2.resize(img, size, interpolation=cv2.INTER_AREA) #resize image
    processed = ut.filter_image(processed, 200, grayscale=True) #filter image
    processed = cv2.normalize(processed, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return processed


def convert_equation_to_latex():

    # get bounding boxes

    # for box in bounding boxes
    #   predict each character 
    #   put in a map 
    #   
    return -1

if __name__ == "__main__":
    main()