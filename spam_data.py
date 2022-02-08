import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('wordnet')


spam_data = pd.read_csv('spam.csv',sep=',',names=['label','messages','dsds','svdsdv','dvb'])

## Data preprocessing

spam_data = spam_data.drop(columns=(['dsds','svdsdv','dvb']))
spam_data = spam_data.drop(index=0)
spam_data = spam_data.reset_index()
spam_data = spam_data.drop(columns=(['index']))
## step 1 : prepare a pipline for text preprocessing

corpus = []
ps = PorterStemmer()
for i in range(0,len(spam_data)):
    review= re.sub('[^a-zA-Z]',' ',spam_data['messages'][i])
    review = review.lower()
    review = review.split()
    stem = [ps.stem(word) for word in review  if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


## step 2 : create a document matrix
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)
x = cv.fit_transform(corpus)

## step 3 : create x and y
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
y = lb.fit_transform(spam_data['label'])

## Model Bulding
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,stratify=y)

## Model Training
from sklearn.naive_bayes import MultinomialNB
NB_classifier = MultinomialNB()
NB_classifier.fit(x_train,y_train)

## Model Testing
y_pred = NB_classifier.predict(x_test)

## Model Evalution
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
accuracy_scr = accuracy_score(y_test,y_pred)
confusion_max = confusion_matrix(y_test,y_pred)
precision = precision_score(y_test,y_pred)

