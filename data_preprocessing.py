"""
Data Preprocessing Script

- enumerates data in the raw_datasets/ and places stardized ouput pickles in the processes_dataset folder
- converts to standard format (jpeg)
- drops unsupported characters
- converts to custom bool encoding format
"""

import os

def main():

    root_dir = "/raw_datasets"

    for folder, _, files in os.walk(root_dir):
        pass

if __name__ == "__main__":
    main()



