import os
import shutil

def move_images(directory):
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.png'):  # Assuming the images have .png extension
            # Extract the first word before the first dash
            first_word = filename.split('-')[0].strip()
            # Create the destination directory if it doesn't exist
            destination_dir = os.path.join(directory, first_word)
            os.makedirs(destination_dir, exist_ok=True)
            # Move the image file to the destination directory
            source_path = os.path.join(directory, filename)
            destination_path = os.path.join(destination_dir, filename)
            shutil.move(source_path, destination_path)
            print(f"Moved {filename} to {destination_dir}")

# Replace 'directory_path' with the path of the directory containing images
# directory_path = '/Users/jibly/Documents/eecs442/cv_final_project/training_symbols/symbols'
# move_images(directory_path)


def combine_directories(parent_dir1, parent_dir2, common_dir):
    # Create the common directory if it doesn't exist
    os.makedirs(common_dir, exist_ok=True)

    # Iterate through subdirectories in the first parent directory
    for subdir1 in os.listdir(parent_dir1):
        # Check if the subdirectory exists in the second parent directory
        subdir2 = os.path.join(parent_dir2, subdir1)
        if os.path.exists(subdir2):
            # Combine the contents of the two subdirectories
            combine_subdirectories(os.path.join(parent_dir1, subdir1), subdir2, os.path.join(common_dir, subdir1))
        else:
            # Move the subdirectory from the first parent directory to the common directory
            shutil.move(os.path.join(parent_dir1, subdir1), os.path.join(common_dir, subdir1))
            print(f"Moved directory {subdir1} from {parent_dir1} to {common_dir}")

    # Move remaining subdirectories from the second parent directory to the common directory
    for subdir2 in os.listdir(parent_dir2):
        if not os.path.exists(os.path.join(common_dir, subdir2)):
            shutil.move(os.path.join(parent_dir2, subdir2), os.path.join(common_dir, subdir2))
            print(f"Moved directory {subdir2} from {parent_dir2} to {common_dir}")

def combine_subdirectories(subdir1, subdir2, common_subdir):
    # Create the common subdirectory if it doesn't exist
    os.makedirs(common_subdir, exist_ok=True)

    # Iterate through files in the first subdirectory
    for item in os.listdir(subdir1):
        # Move files from the first subdirectory to the common subdirectory
        shutil.move(os.path.join(subdir1, item), os.path.join(common_subdir, item))
        print(f"Moved {item} from {subdir1} to {common_subdir}")

    # Iterate through files in the second subdirectory
    for item in os.listdir(subdir2):
        # Move files from the second subdirectory to the common subdirectory
        shutil.move(os.path.join(subdir2, item), os.path.join(common_subdir, item))
        print(f"Moved {item} from {subdir2} to {common_subdir}")

    # # Remove the original subdirectories
    # os.rmdir(subdir1)
    # os.rmdir(subdir2)
    # print(f"Removed directories: {subdir1}, {subdir2}")

# Replace 'parent_directory1' and 'parent_directory2' with the paths of the parent directories
# parent_directory1 = 'training_symbols/symbols'
# parent_directory2 = 'training_symbols/symbols-2'
# # Replace 'common_directory' with the path of the new common directory
# common_directory = 'combined_training'
# combine_directories(parent_directory1, parent_directory2, common_directory)

def rename_files_with_subdirectory_name(source_folder):
    """
    Rename all files within subdirectories to begin with the name of the subdirectory followed by a number.

    Args:
    - source_folder (str): Path to the source folder containing subdirectories with files.
    """
    # Iterate through subdirectories in the source folder
    for root, dirs, files in os.walk(source_folder):
        for dir_name in dirs:
            # Get the path of the subdirectory
            subdir_path = os.path.join(root, dir_name)
            
            # Iterate through files in the subdirectory
            for index, file_name in enumerate(os.listdir(subdir_path)):
                # Construct the new file name
                new_file_name = f"{dir_name}_{index + 1}.jpg"  # You can adjust the extension as needed
                
                # Construct the paths of the original and new files
                old_file_path = os.path.join(subdir_path, file_name)
                new_file_path = os.path.join(subdir_path, new_file_name)
                
                # Rename the file
                os.rename(old_file_path, new_file_path)
                print(f"Renamed {file_name} to {new_file_name}")

# Example usage:
# Replace 'source_folder' with the path of the source folder containing subdirectories with files
source_folder = 'combined_training'

# Call the function to rename files
rename_files_with_subdirectory_name(source_folder)