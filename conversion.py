import torch
import cv2
import utils as ut
import torch.nn as nn

num_classes = 43


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

    for f in fs:
        img = ut.chunk_image_path(f)
        get_box_map(img)

    print("This is a ulitiy module")

#class_names = ['(', ')', '+', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', 'C', 'a', 'b', 'd', 'dot', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'p', 'pi', 'q', 'r', 's', 'slash', 't', 'u', 'v', 'w', 'x', 'y', 'z']
class_names = ['(', ')', '+', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', 'C', 'a', 'b', 'dot', 'e', 'k', 'p', 'pi', 'slash', 'u', 'v', 'w', 'x', 'y', 'z']

30

def predict(img, model):

    img = convert_numpy_array(img)

    img = torch.from_numpy(img)
    img = img.repeat(1, 3, 1, 1)
    outputs = model(img)
    _, preds = torch.max(outputs, 1)
    # print(preds)
    # print(preds[0])
    # print(class_names)
    # exit(1)


    # given an img (32 x 32) return predicted class
    return class_names[preds[0]]


size = (32, 32)
def convert_numpy_array(img):
    
    processed = img
    processed = cv2.resize(img, size, interpolation=cv2.INTER_AREA) #resize image
    processed = ut.filter_image(processed, 200, grayscale=False) #filter image
    processed = cv2.normalize(processed, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return processed

def box_to_class(img, box):
    
    # processed = img
    # processed = cv2.resize(img, size, interpolation=cv2.INTER_AREA) #resize image
    # processed = ut.filter_image(processed, 200, grayscale=True) #filter image
    # processed = cv2.normalize(processed, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # return processed


    pass

def get_box_map(img):

    # 1. conv to grayscale
    # 2. get the boxes
    # 3. loop over boxes and print preductions

    #gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    boxes = ut.grab_bounding_boxes(img)

    model = torch.load("transfer_model_custom_net_clean.pth", map_location=torch.device('cpu'))
    model.eval()

    for box in boxes:
        sub_image = img[box[0]-4:box[1]+4, box[2]-4:box[3]+4]
        c = predict(sub_image, model)

        cv2.imshow(c, sub_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    pass



def convert_equation_to_latex():

    # get bounding boxes

    # for box in bounding boxes
    #   predict each character 
    #   put in a map 
    #   
    return -1

if __name__ == "__main__":
    main()