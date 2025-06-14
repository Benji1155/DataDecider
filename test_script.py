import nltk
nltk.download('punkt', download_dir='nltk_data')

# Force tokenization to trigger creation of all needed files like punkt_tab
from nltk.tokenize import word_tokenize
word_tokenize("Hello there!")  # This forces NLTK to load everything fully
