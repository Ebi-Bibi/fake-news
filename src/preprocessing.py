import re
import nltk
from nltk.corpus import stopwords

# try:
#     nltk.data.find('corpora/stopwords')
# except KeyError:
#     nltk.download('stopwords')

def clean_text(text):
    """
    Fungsi modular untuk membersihkan teks berita dari tanda baca, URL, 
    dan kata umum (stopwords) bahasa Inggris.
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'.*?\(reuters\)\s*-', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    words = text.split()
    cleaned_words = [word for word in words if word not in stop_words]
    
    return " ".join(cleaned_words)