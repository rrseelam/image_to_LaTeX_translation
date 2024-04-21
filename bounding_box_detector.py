"""
Data Preprocessing Script

- enumerates data in the raw_datasets/ and places stardized ouput pickles in the processes_dataset folder
- converts to standard format (jpeg)
- drops unsupported characters
- converts to custom bool encoding format
"""

def main():
    
    print("This is a module that, given an image, returns an array of bounding boxes")

def grab_bounding_boxes(img):
    # returns an np array of c1, c2, c3, c4 which are the corner in clockwise order from TL
    # 
    #   C1   C2
    #
    #   C4   C3
    #
    # image must be an np_array of size H,W with values of 0 or 1 
    pass


if __name__ == "__main__":
    main()