import nltk
import os

# Define the directory where the data will be downloaded
# This will create a folder named 'nltk_data' in your current project directory
download_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nltk_data')

# Create the directory if it doesn't exist
if not os.path.exists(download_dir):
    os.makedirs(download_dir)

# Download the 'punkt' and 'wordnet' packages to the specified directory
print(f"Downloading NLTK data to: {download_dir}")
nltk.download('punkt', download_dir=download_dir)
nltk.download('wordnet', download_dir=download_dir)

print("\n✅ Download complete. You should now have an 'nltk_data' folder in your project directory.")
print("You can now deploy your application with this folder included.")
