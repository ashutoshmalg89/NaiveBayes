from __future__ import division
import os
import glob
import math
import nltk
import sys
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
reload(sys)
sys.setdefaultencoding("utf-8")

tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
stemmer = PorterStemmer()
stopwords = sorted(stopwords.words('english'))
corpus = "F:/MS_OneDriveBkp/Fall16/ML/HW1/train"
test_data = "F:/MS_OneDriveBkp/Fall16/ML/HW1/test"


train_file_dict={}
data_dic={}
bag_of_words = {}
vocabulary=[]
words_in_train=0
conditional_prob={}
class_length={}
accuracy=0.0

def tokenize(doc):
    tokens = tokenizer.tokenize(doc)
    tokens_lower = [token.lower() for token in tokens]
    proc_tokens = [stemmer.stem(token) for token in tokens_lower if not token in stopwords]
    return proc_tokens

def max_val(result):
    return max(result, key=result.get)

def trainNaiveBayes(corpus):
    print "In train"
    unique_words={}
    for labels in os.listdir(corpus):
        list_of_files = glob.glob(corpus + '/' + labels)
        for filepath in list_of_files:
            files = os.listdir(filepath)
            file_array=[]
            for filename in files:
                filedir = filepath + '/' + filename
                file=open(filedir, 'r')
                doc = file.read()
                file.close()
                tokens=tokenize(doc)
                file_array = file_array + tokens

            for word in file_array:
                if word not in unique_words:
                    unique_words[word]=1

            data_dic[labels]=Counter(file_array)
            class_length[labels]=len(file_array)

    return data_dic, class_length,len(unique_words)

def testNaiveBayes(data_dic,class_length, total_words_train ):
    print "In Test"
    correct_predict = 0
    wrong_predict = 0
    file1=[]
    count =0
    for labels in os.listdir(test_data):
        actual_class= labels
        list_of_files = glob.glob(test_data + '/' + labels)
        for filepath in list_of_files:
            files = os.listdir(filepath)
        for filename in files:

            filedir = filepath + '/' + filename
            file = open(filedir, 'r').read()
            file = tokenize(file)
            result={}
            predicted_class=None
            for label in data_dic:
                probability=0
                for word in file:
                    if word in data_dic[label]:
                        probability+=math.log10((data_dic[label][word]+1)/(class_length[label]+total_words_train))
                    else:
                        probability += math.log10((1) / (class_length[label] + total_words_train))

                probability = probability/len(class_length)
                result[label]=probability

            predicted_class=max_val(result)

            if actual_class==predicted_class:
                correct_predict=correct_predict+1
            else:
                wrong_predict = wrong_predict+1
    accuracy=(correct_predict/(correct_predict+wrong_predict)*100)

    return accuracy



if __name__=='__main__':
    data_dic,  class_length, unique_words=trainNaiveBayes(corpus)
    accuracy =testNaiveBayes(data_dic,class_length, unique_words)
    print accuracy
