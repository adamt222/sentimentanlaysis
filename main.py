import time
import streamlit as st
import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from nltk.stem.porter import PorterStemmer
porter=PorterStemmer()
count=CountVectorizer()
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop=stopwords.words('english')


#This will replace html tags with empty strings
import re
def preprocessor(text):
             text=re.sub('<[^>]*>','',text)
             emojis=re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text)
             text=re.sub('[\W]+',' ',text.lower()) +\
                ' '.join(emojis).replace('-','')
             return text

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

tfidf=TfidfVectorizer(strip_accents=None,lowercase=False,preprocessor=None,tokenizer=tokenizer_porter,use_idf=True,norm='l2',smooth_idf=True)


header = st.container()
dataset = st.container()
model_training = st.container()

with header:
    st.title("Welcome to this sentiment analysis app !")
    st.text("In this project I will train a model to try and figure the sentiment of some text out")

with dataset:
    review_data = pd.read_csv('data/IMDBDataset.csv')
    sentiment_dist = pd.DataFrame(review_data['sentiment'].value_counts())
    st.bar_chart(sentiment_dist)

with model_training:
    st.header("Time to train the model!")

    review_data['text'] = review_data['text'].apply(preprocessor)

    y=review_data.sentiment.values
    x=tfidf.fit_transform(review_data.text)

    X_train,X_test,y_train,y_test=train_test_split(x,y,random_state=1,test_size=0.5,shuffle=False)

    print("Abount to train")
    now = time.time()
    clf=LogisticRegressionCV(cv=6,scoring='accuracy',random_state=0,n_jobs=-1,verbose=3,max_iter=500).fit(X_train,y_train)
    later = time.time()
    print("Finished training")
    print("Time taken = ", int(later-now))
    y_pred = clf.predict(X_test)

    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
