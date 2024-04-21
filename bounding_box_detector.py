"""
Data Preprocessing Script

- enumerates data in the raw_datasets/ and places stardized ouput pickles in the processes_dataset folder
- converts to standard format (jpeg)
- drops unsupported characters
- converts to custom bool encoding format
"""

import numpy as np

def main():
    
    print("This is a module that, given an image, returns an array of bounding boxes")


def flood(img, h, w, visited, coor):

    if h < 0 or w < 0:
        return
    
    H, W = img.shape

    if h >= H or w >= W:
        return

    if visited[h,w] == 1:
        return

    if img[h,w] != 0:
        return
    
    visited[h,w] = 1

    coor[0] = min(h, coor[0])
    coor[1] = max(h, coor[1])

    coor[2] = min(w, coor[2])
    coor[3] = max(w, coor[3])

    flood(img, h+1, w, visited, coor)
    flood(img, h-1, w, visited, coor)

    flood(img, h, w-1, visited, coor)
    flood(img, h, w+1, visited, coor)

    flood(img, h+1, w+1, visited, coor)
    flood(img, h+1, w-1, visited, coor)

    flood(img, h-1, w+1, visited, coor)
    flood(img, h-1, w-1, visited, coor)
    
    return coor


def grab_bounding_boxes(img):
    # returns an np array of c1, c2, c3, c4 which are the corner in clockwise order from TL
    # 
    #   C1   C2
    #
    #   C4   C3
    #
    # image must be an np_array of size H,W with values of 0 or 256 (result for util function)

    H, W = img.shape

    visited = np.zeros_like(img)

    res = []

    for h in range(H):
        for w in range(W):
            
            coor = flood(img, h, w, visited, [h, h, w, w])
            
            if coor == None:
                continue

            if coor == [h, h, w, w]:
                continue

            if coor[1] - coor[0] < 16:
                continue

            if coor[3] - coor[2] < 16:
                continue

            res.append(coor)

    return res
    

if __name__ == "__main__":
    main()