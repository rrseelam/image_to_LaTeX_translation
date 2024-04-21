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


def correct_for_equal(res):
    
    real_res = []
    print(res)
    ignore = set()

    for i in range(len(res)):
        
        if i in ignore:
            continue

        if abs(res[i][1] - res[i][0]) > 8:

            real_res.append(res[i])
            continue

        for j in range(i+1, len(res)):

            

            if j in ignore:
                continue


            if abs(res[i][2] - res[j][2]) < 10 and abs(res[i][3] - res[j][3]) < 10:
                
                ignore.add(i)
                ignore.add(j)

                lo = min(res[i][0], res[j][0])
                hi = max(res[i][1], res[j][1])

                l =  min(res[i][2], res[j][2])
                r =  max(res[i][3], res[j][3])

                c = [lo, hi, l , r]

                real_res.append(c)
                break
   
    return real_res





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

            if coor[1] - coor[0] < 4:
                continue

            if coor[3] - coor[2] < 4:
                continue

            res.append(coor)

    res = correct_for_equal(res)

    return res
    

if __name__ == "__main__":
    main()