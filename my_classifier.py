# Load libraries
import numpy as np
import sys
import pandas as pd
import re
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
# from yellowbrick.classifier import ClassificationReport
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score,classification_report
from sklearn import tree

train_file = sys.argv[1]
test_file = sys.argv[2]

dFrame_train_file = pd.read_csv(train_file, sep = '\t', names=['instance','text','topic'])
dFrame_test_file = pd.read_csv(test_file, sep = '\t', names=['instance','text','topic'])

trainStr = np.array(dFrame_train_file['text'])
testStr= np.array(dFrame_test_file['text'])
testNum = np.array(dFrame_test_file['instance'])
trainTopic = np.array(dFrame_train_file['topic'])
testTopic = np.array(dFrame_test_file['topic'])

def predict_and_test(model, X_test_bag_of_words):
    predicted_y = model.predict(X_test_bag_of_words)
    # for i in range(0,len(testStr)):
        # print(testNum[i], predicted_y[i])
    print(classification_report(testTopic, predicted_y,zero_division=0))

def preprocessing(text):

    trainStr = text
    valid_sentence =[]
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    word_pattern = re.compile(r'[^#@_$%\sa-zA-Z\d]')

    for words in trainStr:
        url_match = url_pattern.search(words)
        symbol_match = word_pattern.search(words)

        #removing words less than 2 characters in length 
        words = re.sub(r'\b\w{,1}\b', '', words)
        if url_match:
            without_url = re.sub(url_pattern,' ',words)
            valid_sentence.append(without_url)
        #checks if symbol is other than the accepted ones 
        elif symbol_match:

            without_symbol = re.sub(word_pattern,'',words)
            valid_sentence.append(without_symbol)
        else:
            valid_sentence.append(words)
            valid_sentence.append(words)

    valid_sentence = np.array(valid_sentence)
    clean_data = np.array(valid_sentence)
    return clean_data


train_data = preprocessing(trainStr)
test_data = preprocessing(testStr)

# print(train_data.shape)
# print(test_data.shape)

# count = CountVectorizer(lowercase=False,token_pattern='[#@_$%\w\d]{2,}')
count = CountVectorizer(lowercase=False,max_features=1400,token_pattern ='[#@_$%\w\d]{2,}', stop_words='english')
X_train_bag_of_words = count.fit_transform(train_data)

#test
X_test_bag_of_words = count.transform(test_data)

clf = MultinomialNB()
model = clf.fit(X_train_bag_of_words, trainTopic)
predict_and_test(model, X_test_bag_of_words)