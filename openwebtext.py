import os
import tarfile
import lzma
import shutil

"""
This code is for extracting the text from openwebtext from huggingface dataset Skylion007/openwebtext
Check this out: https://huggingface.co/datasets/Skylion007/openwebtext
"""


# Set up paths
base_path = 'openwebtext/subsets'
output_file = 'openwebtext_dataset.txt'

n_files = 0
# Process tar files in the subsets folder
with open(output_file, 'w', encoding='utf-8') as outfile:
    for tar_file in os.listdir(base_path):
        tar_path = os.path.join(base_path, tar_file)
        
        # Check if it's a tar file
        if tar_file.endswith('.tar'):
            print(f"Processing tar file: {tar_file}")
            
            # Create a temporary directory for extraction
            temp_dir = os.path.join(base_path, 'temp_extract')
            os.makedirs(temp_dir, exist_ok=True)
            
            # Extract the tar file
            with tarfile.open(tar_path, 'r') as tar:
                tar.extractall(path=temp_dir)
            
            # Process the extracted 'openwebtext' folder
            openwebtext_path = os.path.join(temp_dir, 'openwebtext')
            
            # Process .xz files in the 'openwebtext' folder
            for xz_file in os.listdir(openwebtext_path):
                if xz_file.endswith('.xz'):
                    xz_path = os.path.join(openwebtext_path, xz_file)
                    print(f"Processing .xz file: {xz_file}")
                    
                    # Decompress the .xz file
                    with lzma.open(xz_path, 'rb') as xz:
                        # xz file contains a tar archive
                        with tarfile.open(fileobj=xz) as tar:
                            # Extract and concatenate all files from this inner tar archive
                            for member in tar.getmembers():
                                if member.isfile():  # Ensure it's a file, not a directory
                                    f = tar.extractfile(member)
                                    if f:
                                        content = f.read().decode('utf-8')
                                        outfile.write(content)
                                        outfile.write('\n\n')  # Add some separation between files
                                        n_files += 1
                                else:
                                    print(f"Skipping non-file: {member.name}")
                else:
                    print(f"Skipping non-.xz file: {xz_file}")
            
            # Clean up the temporary directory
            shutil.rmtree(temp_dir)

print(f"{n_files} .txt files have been concatenated into {output_file}")
