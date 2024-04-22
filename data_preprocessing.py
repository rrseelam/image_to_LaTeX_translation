"""
Data Preprocessing Script

- enumerates data in the raw_datasets/ and places stardized ouput pickles in the processes_dataset folder
- converts to standard format (jpeg)
- drops unsupported characters
- converts to custom bool encoding format
"""

import os
import utils as ut
import cv2
import numpy as np
from matplotlib import pyplot as plt


def process_img(img, lb, ub, size=(32,32)):
    processed = cv2.resize(img, size, interpolation=cv2.INTER_AREA) #resize image
    processed = ut.filter_image(processed, lb, ub, grayscale=True) #filter image
    return processed

def process_all(root_dir, output_dir):
    for root, dirs, files in os.walk(root_dir):
        for dir_name in dirs:
            source_subdir = os.path.join(root, dir_name)
            destination_subdir = os.path.join(output_dir, dir_name)

            # Create the destination subdirectory if it doesn't exist
            os.makedirs(destination_subdir, exist_ok=True)

            # Process images in the source subdirectory and copy them to the destination subdirectory
            for file_name in os.listdir(source_subdir):
                source_file_path = os.path.join(source_subdir, file_name)
                destination_file_path = os.path.join(destination_subdir, file_name)
                
                # Read the image
                image = cv2.imread(source_file_path)
                
                # Call the process function on the image
                processed_image = process_img(image, 200, 200)
                
                # Save the processed image to the destination subdirectory
                cv2.imwrite(destination_file_path, processed_image)
                print(f"Processed and copied {file_name} to {destination_subdir}")

def save_as_numpy(root_dir, output_dir):
    images = []
    labels = []

    # Iterate through subdirectories in the parent directory
    for _, subdirectory in enumerate(os.listdir(root_dir)):
        subdirectory_path = os.path.join(root_dir, subdirectory)
        if os.path.isdir(subdirectory_path):
            # Iterate through files in the subdirectory
            for file_name in os.listdir(subdirectory_path):
                # Read the image
                image_path = os.path.join(subdirectory_path, file_name)
                image = cv2.imread(image_path)

                # Append the image to the list
                images.append(image)
                # Append the label (subdirectory name) to the list
                labels.append(subdirectory)

                print(f"Processed {file_name} in {subdirectory}")

    # Convert the lists to numpy arrays
    images_array = np.array(images)
    labels_array = np.array(labels)

    # Save the numpy arrays to the output file
    np.savez(output_dir, images=images_array, labels=labels_array)
    print(f"Saved images and labels to {output_dir}")

def main():
    # process_all(root_dir="combined_training", output_dir="processed_training")
    # save_as_numpy(root_dir="processed_training", output_dir="datasets/symbols.npz")

    for i in range(10):
        data = np.load('datasets/symbols.npz')
        imgs = data['images']
        labels = data['labels']

        rand_idx = np.random.randint(0, imgs.shape[0])  
        plt.clf()
        plt.imshow(imgs[rand_idx], interpolation='nearest')
        plt.title("label = " + labels[rand_idx])
        plt.show()


if __name__ == "__main__":
    main()



