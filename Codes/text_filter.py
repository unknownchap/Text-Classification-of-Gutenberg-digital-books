import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words("english"))
lemmatiser = WordNetLemmatizer()


def text_filter(text):
    # 1. Removal of Punctuation Marks 
    nopunct = [char for char in text if char not in string.punctuation]
    nopunct = ''.join(nopunct)
    # 2. Lemmatisation 
    lemmitized = ''
    i = 0
    for i in range(len(nopunct.split())):
        x = lemmatiser.lemmatize(nopunct.split()[i], pos="v")
        lemmitized = lemmitized + x + ' '
    # 3. Removal of Stopwords
    noStopWords = [word for word in lemmitized.split() if word.lower() not 
            in stopwords.words('english')]
    return " ".join(noStopWords)
