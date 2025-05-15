import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from langdetect import detect

# You'll need to initialize NLTK and download resources at startup
def initialize_nltk():
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

# Emoji removal pattern
emoji_pattern = re.compile(
    "["
    u"\U0001F600-\U0001F64F"  # Emoticons
    u"\U0001F300-\U0001F5FF"  # Symbols & Pictographs
    u"\U0001F680-\U0001F6FF"  # Transport & Map
    u"\U0001F1E0-\U0001F1FF"  # Flags
    u"\U00002700-\U000027BF"  # Dingbats
    u"\U0001F900-\U0001F9FF"  # Supplemental symbols
    u"\U0001FA70-\U0001FAFF"  # Extended-A (🫵)
    u"\U00002600-\U000026FF"  # Misc symbols
    u"\U0001F000-\U0001F02F"  # Mahjong
    u"\U0000203C"             # Double exclamation
    u"\uFE0F"                 # Variation Selector-16
    "]+",
    flags=re.UNICODE
)

# Custom punctuation (including Bangla, Urdu, Arabic)
custom_punct = '।॥''""…،؛؟'
all_punct = string.punctuation + custom_punct

# Detect Bengali characters
def is_bengali(word):
    return bool(re.search(r'[\u0980-\u09FF]', word))

# Clean and preprocess text (simplified version for deployment)
def preprocess_text(text):
    # Initialize lemmatizer and stopwords
    en_lem = WordNetLemmatizer()
    en_stop = set(stopwords.words('english'))
    
    text = str(text)

    # Remove URLs, hashtags, mentions
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"@\w+", "", text)

    # Remove emojis
    text = emoji_pattern.sub('', text)

    # Remove punctuations
    text = re.sub(f"[{re.escape(all_punct)}]", "", text)

    # Normalize space
    text = re.sub(r'\s+', ' ', text).strip()

    try:
        lang = detect(text)
    except:
        lang = 'en'

    # For English, use NLTK
    # Tokenize into words
    tokens = word_tokenize(text)
    tokens = [t.lower() for t in tokens if t.isalpha()]

    clean_tokens = []
    for token in tokens:
        if is_bengali(token):
            # Keep Bengali tokens as is
            clean_tokens.append(token)
        else:
            if token not in en_stop:
                lemma = en_lem.lemmatize(token)
                clean_tokens.append(lemma)

    return ' '.join(clean_tokens)  # Return as space-separated string for TF-IDF