import os
import inkml2img

cwd = os.getcwd()

input_dir = cwd + "/archive/TrainINKML_2013"
output_dir = cwd + "/archive/TrainINKML_2013-converted"

# Get a list of all files in the input directory
files = os.listdir(input_dir)

# Filter the list to include only .inkml files
inkml_files = [file for file in files if file.endswith('.inkml')]

# Process each .inkml file
for file in inkml_files:
    input_file = os.path.join(input_dir, file)
    output_file = os.path.join(output_dir, file.replace('.inkml', '.png'))
    inkml2img.inkml2img(input_file, output_file)