import os
import shutil
import glob

def move_png_files_to_results():
    """
    Move all PNG files from the project root to the results/plots directory
    """
    # Create the results/plots directory if it doesn't exist
    os.makedirs("results/plots", exist_ok=True)
    
    # Find all PNG files in the project root
    png_files = glob.glob("*.png")
    
    if not png_files:
        print("No PNG files found in the project root directory.")
        return
    
    print(f"Found {len(png_files)} PNG files in the project root directory.")
    
    # Move each file to the results/plots directory
    for png_file in png_files:
        destination = os.path.join("results/plots", png_file)
        shutil.move(png_file, destination)
        print(f"Moved {png_file} to {destination}")
    
    print("All PNG files have been moved to the results/plots directory.")

if __name__ == "__main__":
    move_png_files_to_results()
