import os

def pad_filename_with_zeros(directory):
    # Walk through the directory and its subdirectories
    for root, _, files in os.walk(directory):
        for file in files:
            # Check if the file is a .jpg file
            if file.endswith('.jpg'):
                # Get the full path of the file
                full_path = os.path.join(root, file)
                # Extract the filename without the extension
                filename, ext = os.path.splitext(file)
                
                # Pad the filename with zeros to make it 5 digits
                if filename.isdigit():  # Only pad if filename is numeric
                    new_filename = filename.zfill(5) + ext
                    new_full_path = os.path.join(root, new_filename)
                    
                    # Rename the file
                    os.rename(full_path, new_full_path)
                    print(f'Renamed {full_path} to {new_full_path}')

# Replace 'your_directory_path' with the path to your directory
pad_filename_with_zeros('/dataset/AIC2024/pumkin_dataset/0/frames/pyscenedetect_t_5_fix')