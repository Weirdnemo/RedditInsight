import re
from typing import Optional
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

download_nltk_data()

# Custom stopwords list
CUSTOM_STOPWORDS = {
    'http', 'https', 'www', 'com', 'org', 'net', 'imgur', 'jpg', 'png', 'gif', 'webp', 'amp',
    'reddit', 'redd', 'edit', 'deleted', 'removed', 'the', 'and', 'that', 'this', 'but', 'they',
    'have', 'from', 'what', 'when', 'where', 'which', 'who', 'why', 'how', 'all', 'any', 'both',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'than', 'too', 'very', 'can', 'will',
    'just', 'should', 'now'
}

def clean_text_for_wordcloud(text: str) -> str:
    """
    Clean text for word cloud generation.
    
    Args:
        text (str): Input text to clean
        
    Returns:
        str: Cleaned text
    """
    try:
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize and filter words
        words = word_tokenize(text)
        filtered_words = [
            word for word in words 
            if word not in CUSTOM_STOPWORDS 
            and len(word) > 2
        ]
        
        return ' '.join(filtered_words)
    except Exception as e:
        print(f"Error cleaning text: {e}")
        return ""

def extract_thread_id(url: str) -> Optional[str]:
    """
    Extract thread ID from Reddit URL.
    
    Args:
        url (str): Reddit URL
        
    Returns:
        Optional[str]: Thread ID or None if not found
    """
    try:
        # Extract thread ID from URL
        match = re.search(r'reddit\.com/r/[^/]+/comments/(\w+)/', url)
        if match:
            return match.group(1)
        return None
    except Exception as e:
        print(f"Error extracting thread ID: {e}")
        return None
