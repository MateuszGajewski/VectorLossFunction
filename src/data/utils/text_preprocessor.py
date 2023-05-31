import string
from nltk.corpus import stopwords
import re
import nltk


class TextPreprocessor:
    def __init__(self):
        self.stemmer = nltk.SnowballStemmer("english")
        stop_words = stopwords.words('english')
        more_stopwords = ['u', 'im', 'c']
        self.stop_words = stop_words + more_stopwords

    def preprocess(self, df):
        df['clean_text'] = df['short_description'].apply(self.clean_text)
        df['clean_text'] = df['clean_text'].apply(self.remove_stopwords)
        df['clean_text'] = df['clean_text'].apply(self.stemm_text)
        df['clean_text'] = df['clean_text'].apply(self.preprocess_data)
        return df

    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)
        return text

    def remove_stopwords(self, text):
        words = text.split(' ')
        words = [word for word in words if word not in self.stop_words]
        text = ' '.join(words)
        return text

    def stemm_text(self, text):
        text = ' '.join(self.stemmer.stem(word) for word in text.split(' '))
        return text

    def preprocess_data(self, text):
        text = self.clean_text(text)
        text = ' '.join(word for word in text.split(' ') if word not in self.stop_words)
        text = ' '.join(self.stemmer.stem(word) for word in text.split(' '))
        return text

