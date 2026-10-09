import os
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def load_text_file(file_path):
    """Safely opens and reads a text file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at {file_path} was not found.")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def preprocess_text(raw_text):
    """Applies basic text cleaning operations."""
    text = raw_text.lower()
    
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    tokens = word_tokenize(text)
    
    
    return tokens

def save_processed_tokens(tokens, output_path):
    """Saves the cleaned tokens into a new text file."""
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(" ".join(tokens))

# --- Example Usage ---
if __name__ == "__main__":
    input_file = "sample.txt"
    output_file = "cleaned_output.txt"
    
    # Create a dummy file for demonstration purposes
    with open(input_file, 'w', encoding='utf-8') as f:
        f.write("Test Test Tes.")
        
    try:
        # Run pipeline
        print("Reading file...")
        raw_data = load_text_file(input_file)
        
        print("Preprocessing text...")
        processed_words = preprocess_text(raw_data)
        
        print("Saving cleaned text...")
        save_processed_tokens(processed_words, output_file)
        
        print(f"\nSuccess! Processed output: {processed_words}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
