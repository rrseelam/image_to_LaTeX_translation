import torch
import cv2
import utils as ut
import torch.nn as nn
import torchvision
from torchvision import models
import torch.nn.functional as F
import numpy as np

#class_names = ['(', ')', '+', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', 'C', 'a', 'b', 'd', 'dot', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'p', 'pi', 'q', 'r', 's', 'slash', 't', 'u', 'v', 'w', 'x', 'y', 'z']
#class_names = ['(', ')', '+', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', 'C', 'a', 'b', 'dot', 'e', 'k', 'p', 'pi', 'slash', 'u', 'v', 'w', 'x', 'y', 'z']
#class_names = ['(', ')', '+', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', 'C', 'a', 'b', 'dot', 'e', 'i', 'p', 'pi', 'slash', 'u', 'x', 'y']
class_names = ['(', ')', '+', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', 'a', 'b', 'd', 'dot', 'e', 'p', 'pi', 'slash', 'u', 'x', 'y']

num_classes = len(class_names)


model_ft = models.wide_resnet101_2(pretrained=True)
num_ftrs = model_ft.fc.in_features

half_in_size = round(num_ftrs/2)
layer_width = 1024

class SpinalNet(nn.Module):
    def __init__(self):
        super(SpinalNet, self).__init__()
        
        self.fc_spinal_layer1 = nn.Sequential(
            nn.Dropout(p = 0.5), nn.Linear(half_in_size, layer_width),
            nn.BatchNorm1d(layer_width), nn.ReLU(inplace=True),)
        self.fc_spinal_layer2 = nn.Sequential(
            nn.Dropout(p = 0.5), nn.Linear(half_in_size+layer_width, layer_width),
            nn.BatchNorm1d(layer_width), nn.ReLU(inplace=True),)
        self.fc_spinal_layer3 = nn.Sequential(
            nn.Dropout(p = 0.5), nn.Linear(half_in_size+layer_width, layer_width),
            nn.BatchNorm1d(layer_width), nn.ReLU(inplace=True),)
        self.fc_spinal_layer4 = nn.Sequential(
            nn.Dropout(p = 0.5), nn.Linear(half_in_size+layer_width, layer_width),
            nn.BatchNorm1d(layer_width), nn.ReLU(inplace=True),)
        self.fc_out = nn.Sequential(
            nn.Dropout(p = 0.5), nn.Linear(layer_width*4, num_classes),)
        
        

    def forward(self, x):
        x1 = self.fc_spinal_layer1(x[:, 0:half_in_size])
        x2 = self.fc_spinal_layer2(torch.cat([ x[:,half_in_size:2*half_in_size], x1], dim=1))
        x3 = self.fc_spinal_layer3(torch.cat([ x[:,0:half_in_size], x2], dim=1))
        x4 = self.fc_spinal_layer4(torch.cat([ x[:,half_in_size:2*half_in_size], x3], dim=1))
        
        
        x = torch.cat([x1, x2], dim=1)
        x = torch.cat([x, x3], dim=1)
        x = torch.cat([x, x4], dim=1)

        
        x = self.fc_out(x)
        return x


net_fc = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )


class CIFAR10Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3,3), stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)
 
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
 
        self.flat = nn.Flatten()
 
        self.fc3 = nn.Linear(8192, 512)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)
 
        self.fc4 = nn.Linear(512, num_classes)
 
    def forward(self, x):
        # input 3x32x32, output 32x32x32
        x = self.act1(self.conv1(x))
        x = self.drop1(x)
        # input 32x32x32, output 32x32x32
        x = self.act2(self.conv2(x))
        # input 32x32x32, output 32x16x16
        x = self.pool2(x)
        # input 32x16x16, output 8192
        x = self.flat(x)
        # input 8192, output 512
        x = self.act3(self.fc3(x))
        x = self.drop3(x)
        # input 512, output 10
        x = self.fc4(x)
        return x

def main():
    
    fs = ["raw_datasets/CROHME_test_2011-converted/TestData2_2_sub_14.png",
          "raw_datasets/CROHME_test_2011-converted/TestData2_1_sub_2.png",
          "raw_datasets/CROHME_test_2011-converted/TestData2_3_sub_8.png"]
    
    # fs = [f"img{i}.png" for i in range(1,6)]

    for f in fs:

        img = ut.chunk_image_path(f)
        bm = get_box_map(img)
        latex = bm_to_latex(bm)

        cv2.imshow(latex, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()        

    print("This is a ulitiy module")


def predict(img, model, wait=True):

    img = convert_numpy_array(img)
    img_show = img
    img = torch.from_numpy(img)
    
    img = img.repeat(1, 3, 1, 1)
    outputs = model(img)
    _, preds = torch.max(outputs, 1)
    # print(preds)
    # print(preds[0])
    # print(class_names)
    # exit(1)

    # if wait:
    #     cv2.imshow(class_names[preds[0]], img_show)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    # given an img (32 x 32) return predicted class

    print(preds)
    if preds.size == 0:
        return 0
    return class_names[preds[0]]


size = (32, 32)
def convert_numpy_array(img):
    
    processed = img
    processed = cv2.resize(img, size, interpolation=cv2.INTER_AREA) #resize image
    processed = ut.filter_image(processed, 200, grayscale=False) #filter image
    processed = cv2.normalize(processed, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return processed

def custom_compare(x):
    return x[0]

def bm_to_latex(bm):
    
    loc_to_char = []

    # converting to point-char form
    for entry in bm:
        x = (entry[0][2] + entry[0][3]) / 2
        y = (entry[0][0] + entry[0][1]) / 2
        loc_to_char.append((x, y, entry[1]))

    # sort the list based on x position
    loc_to_char = sorted(loc_to_char, key=custom_compare)

    latex = "$" + loc_to_char[0][2]

    main_line = loc_to_char[0][1]
    delta = 15

    runningSup = False
    runningSub = False

    for i in range(1, len(loc_to_char)):
        y = loc_to_char[i][1]   
        if y > main_line + delta:
            # sub

            if runningSup:
                latex += "}"
                runningSup = False
            
            if not runningSub:
                latex += "_{"

            latex += loc_to_char[i][2]

            runningSub = True

        elif y < main_line - delta:
            # sup 

            if runningSub:
                latex += "}"
                runningSub = False

            if not runningSup:
                latex += "^{"

            latex += loc_to_char[i][2]

            runningSup = True

        else:
            if runningSup or runningSub:
                latex += "}"
            runningSup = False
            runningSub = False
            main_line = loc_to_char[i][1]
            latex += loc_to_char[i][2]

    if runningSup or runningSub:
        latex += "}"

    latex += "$"

    latex = latex.replace("dot", ".")
    return latex

def get_box_map(img, wait=False):

    # 1. conv to grayscale
    # 2. get the boxes
    # 3. loop over boxes and print preductions

    #gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    boxes = ut.grab_bounding_boxes(img)

    box_to_char = []

    model = torch.load("transfer_model_spiral_super_clean.pth", map_location=torch.device('cpu'))
    model.eval()
    data_pad = 3
    np_pad = 7
    for box in boxes:

        sub_image = img[box[0] - data_pad:box[1] + data_pad, box[2] - data_pad:box[3] + data_pad]
        sub_image = np.pad(sub_image, pad_width=np_pad, mode='constant', constant_values=255)
        c = predict(sub_image, model)


        if wait:
            cv2.imshow(c, sub_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        box_to_char.append([box, c])
    
    return box_to_char


if __name__ == "__main__":
    main()