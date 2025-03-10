# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 13:48:53 2025

@author: Eleonora Visnevskyte 793917
"""

import pandas
import string
import numpy
import nltk
import random
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from tkinter import *

def function1(gotten_text):
    email_headers = []
    #nltk.download('stopwords')
    dataset = pandas.read_csv('spam_ham_dataset.csv')
    dataset['text'] = dataset['text'].apply(lambda x: x.replace('\r\n', ' '))
    stemmer = PorterStemmer()
    corpus = []
    stopwords_set = set(stopwords.words('english'))
    
    for i in range(len(dataset)):
        all_text = dataset.text.values[i]
        email_headers.append(all_text[:70])
        text = dataset['text'].iloc[i].lower()
        text = text.translate(str.maketrans('', '', string.punctuation)).split()
        text = [stemmer.stem(word) for word in text if word not in stopwords_set]
        text = ' '.join(text)
        corpus.append(text)
    
    vectorizer = CountVectorizer()
    x_data = vectorizer.fit_transform(corpus).toarray()
    y_data = dataset.label_num
    x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(x_data, y_data, test_size = 0.2)
    classifier = RandomForestClassifier(n_jobs=-1)
    classifier.fit(x_data_train, y_data_train)
    #print("Accuracy =", classifier.score(x_data_test, y_data_test))
    
    email_to_classify = gotten_text
    email_text = email_to_classify.lower().translate(str.maketrans('', '', string.punctuation)).split()
    email_text = [stemmer.stem(word) for word in email_text if word not in stopwords_set]
    email_text = ' '.join(email_text)
    email_corpus = [email_text]
    x_email_data = vectorizer.transform(email_corpus)
    if int(classifier.predict(x_email_data)) == 1:
        return "\nThis email is likely to be a scam"
    else:
        return "\nThis email is not likely to be a scam"
    
            
def function2(gotten_text):
    urls_data = pandas.read_csv("urldata12.csv")    
    
    def makeTokens(f):
        tkns_by_slash = str(f.encode('utf-8')).split('/')	 
        total_tokens = []
        for i in tkns_by_slash:
            tokens = str(i).split('-')	
            tkns_by_dot = []
            for j in range(0,len(tokens)):
                temp_tokens = str(tokens[j]).split('.')	
                tkns_by_dot = tkns_by_dot + temp_tokens
            total_tokens = total_tokens + tokens + tkns_by_dot
        total_tokens = list(set(total_tokens))	
        if 'com' in total_tokens:
            total_tokens.remove('com')	
        return total_tokens
    Y = urls_data["label"]
    url_list = urls_data["url"]
    vectorizer = TfidfVectorizer(tokenizer=makeTokens)
    X = vectorizer.fit_transform(url_list)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)	
    logit = LogisticRegression()	
    logit.fit(X_train, Y_train)
    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=400000, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)

    X_predict = [gotten_text]
    X_predict = vectorizer.transform(X_predict)
    new_predict = logit.predict(X_predict)
    if new_predict[0] == "good":
        return "\nThis link is likely to be valid"
    else:    
        return "\nThis link is likely to be dangerous"
    
def function3(gotten_text):
    bad_words = ["immediately", "dear customer", "buy", "urgent", "pay", "win", "lottery", "unpaid"]
    check_marks = 0
    text = gotten_text
    sentence_counter = text.count('.') + text.count('?') + 1
    excl_counter = text.count('!')
    excl_factor = excl_counter/sentence_counter
    caps_counter = sum(1 for c in text if c.isupper())
    caps_factor = caps_counter/sentence_counter
    text = text.lower()
    scam_factor = 0
    for word in bad_words:
        if word in text:
            scam_factor += 1
    if excl_factor > 2:
        check_marks += 1
    if caps_factor > 3:
        check_marks += 1
    if scam_factor > 1:
        check_marks += 1
    
    if check_marks < 1:
        strr = "\nThe text contains "+ str(check_marks) +" out of 3 signs of scam/phishing. It is not likely to be dangerous"
    elif check_marks == 1:
        strr = "\nThe text contains "+ str(check_marks) +" out of 3 signs of scam/phishing. It is not likely to be dangerous, but be aware"
    else:
        strr = "\nThe text contains "+ str(check_marks) +" out of 3 signs of scam/phishing. It is likely a scam/phishing.\nPlease, delete, ignore, or complain about the message."
    return strr

        
global num, txt     
def get_num():
    global num,txt
    num = int(e1.get())
    
def get_txt():
    global num, txt
    txt = str(e2.get())
    if num == 1:
        result = function1(txt)
        
    elif num == 2:
        result = function2(txt)
        
    elif num == 3:
        result = function3(txt)
        
    empty_label = Label(root, text=result)
    empty_label.pack()
    
second = Tk()
second.title('List of suggestions')
second.geometry("700x450")
suggestions = Text(second, height=28, width=100)
suggestions.insert(END, "Here is the list of suggestions for every function:\n\n1) Subject: aep transition items attached is a brief memo outline some of the transtion issues with hpl to aep this is the first draft . the itilized items currently require some more action . please add any items and forward back to me . i will update thanks bob\n\nSubject: looking for medication ? we ` re the best source .\n it is difficult to make our material condition better by the best law , \nbut it is easy enough to ruin it by bad laws . excuse me . . . : ) \nyou just found the best and simpliest site for medication on the net . no perscription , \neasy delivery . private , secure , and easy . better see rightly on a pound a week \nthan squint on a million . we ` ve got anything that you will ever want .  treatment pills ,\n anti - depressant pills , weight loss , and more !\n http : / / splicings . bombahakcx . com / 3 / knowledge and human power are synonymous .\n only high - quality stuff for low rates ! 100 % moneyback guarantee ! there is no god , \nnature sufficeth unto herself in no wise hath she need of an author .\n\n2) pakistanifacebookforever.com/getpassword.php\nwww.radsport-voggel.de/wp-admin/includes/log.exe\n\n3) Happy birthday, Lin! I wish you all the best and I hope we'll meet soon.\n\nURGENT!!!!!!! Pay your bill so that we can deliver your package.")
suggestions.pack()

root = Tk(screenName=None, baseName=None, className='Tk', useTk=1)
root.title("Email Protection App")
root.geometry("500x350")

w = Text(root, height = 5, width = 70)
w.insert(END,'Welcome to Email Protection App\nIt offers choices as\n1) Spam email detection using Machine Learning\n2)Phishing link detection using Machine Learning\n3) Email content analysis')
w.pack()
lab = Label(root, text="\nPlease enter your option below (1, 2 or 3)")
lab.pack()
e1 = Entry(root)
e1.pack()
button = Button(root, command=get_num ,text='Enter', width=16)
button.pack()
w1 = Label(root, text='\nPlease enter your text below (and be patient)')
w1.pack()
e2 = Entry(root)
e2.pack()
button2 = Button(root, command=get_txt ,text='Enter', width=16)
button2.pack()

second.mainloop()
root.mainloop()     
