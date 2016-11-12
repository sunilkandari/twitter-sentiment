import re
import sys
import random
from processTweets import *
from sklearn.multiclass import OneVsOneClassifier
from sklearn import svm
from sklearn import cross_validation
from scipy.sparse import *
import numpy as np
from sklearn.utils import shuffle
sys.path.insert(0,'ark-tokenizer')
from ark import tokenizeRawTweetText

fp = open("preprocessedTweets.txt", 'r')
fp_tr=open("trainpreprocessedTweets.txt", 'w+')
fp_te=open("testpreprocessedTweets.txt", 'w+')
line=fp.readline()

ttc=0
while line:
    rnd=random.random()
    if rnd<0.99 :
        fp_te.write(line)
        ttc+=1
    else:
        fp_tr.write(line)
    line=fp.readline()

print "Total TestTweets %d" %(ttc)

fp_tr.close()
fp_te.close()

fp = open("trainpreprocessedTweets.txt", 'r')

#Create dictionary of words

wordict={}

pos_count=0
neg_count=0
neu_count=0

line=fp.readline()
while line:
    line=line.rstrip()
    fields=re.split(r'\t+',line)
    if len(fields) < 2:
        line=fp.readline()
        continue
    if "positive" == fields[0]:
        pos_count+=1
    elif "negative" == fields[0]:
        neg_count+=1
    else:
        neu_count+=1
        
    tokens=re.split(' ',fields[1])
    size=len(tokens)
    for i in range(size):
        wordict[tokens[i]]=0
    line=fp.readline()

wordlist=sorted(wordict)
wordcount=0
for word in wordlist:
    wordict[word]=wordcount;
    wordcount+=1

fp.close()

wordlist = []
#print pos_count+neg_count+neu_count

#create boolean matrix (no. of tweets)*(no. of words in dict)


#pos_matrix = [[0 for i in range(wordcount)] for j in range(pos_count)]
#neg_matrix = [[0 for i in range(wordcount)] for j in range(neg_count)]
#neu_matrix = [[0 for i in range(wordcount)] for j in range(neu_count)]
pos_matrix = dok_matrix((pos_count,wordcount))
neg_matrix = dok_matrix((neg_count,wordcount))
neu_matrix = dok_matrix((neu_count,wordcount))
fp = open("trainpreprocessedTweets.txt", 'r')
line=fp.readline()

pos=0
neg=0
neu=0

while line:
    line=line.rstrip()
    fields=re.split(r'\t+',line)
    if len(fields) <2:
        line=fp.readline()
        continue
    
    tokens=re.split(' ',fields[1])
    
    size=len(tokens)
    
    if "positive"==fields[0]:
        for i in range(size):
            pos_matrix[pos,wordict[tokens[i]]]=1
        pos+=1

    elif "negative"==fields[0]:
        for i in range(size):
            neg_matrix[neg,wordict[tokens[i]]]=1
        neg+=1

    else:
        for i in range(size):
            neu_matrix[neu,wordict[tokens[i]]]=1
        neu+=1

    line=fp.readline()

pos_matrix.tocsr()
neg_matrix.tocsr()
neu_matrix.tocsr()

pos_matrix = hstack([pos_matrix,csr_matrix([[0],]*pos_count)])
neg_matrix = hstack([neg_matrix,csr_matrix([[1],]*neg_count)])
neu_matrix = hstack([neu_matrix,csr_matrix([[2],]*neu_count)])
final_matrix = vstack([pos_matrix,neg_matrix])
final_matrix = vstack([final_matrix,neu_matrix])
final_matrix = shuffle(final_matrix)
train_Y = final_matrix[:,-1].toarray()[:,0]
print train_Y

print "shape",final_matrix.get_shape()

total_tweets=pos_count+neg_count+neu_count




word_dict = {}
word_list = []
#train_X,train_Y = parse_to_classifier()
#trained_clf = train_classifier(LinearSVC(random_state=0),train_X,train_Y)
score = cross_validation.cross_val_score(OneVsOneClassifier(svm.LinearSVC(random_state=0)),final_matrix[:,:-1],train_Y,cv=5)
print "average accuracy of svm ",score.mean()
#classifying
    
pos_prob=float(pos_count)/float(total_tweets)
neg_prob=float(neg_count)/float(total_tweets)
neu_prob=float(neu_count)/float(total_tweets)

#Create wordnet dictionary
fp_wn=open("wordnet.txt",'r')

wn_dict={}
line=fp_wn.readline()
while line:
    line=line.rstrip()
    fields=line.split(":")
    if fields[0] not in wn_dict:
        wn_dict[fields[0]]=float(fields[1])
    line=fp_wn.readline()

fp_wn.close()

fp = open("testpreprocessedTweets.txt", 'r')
line=fp.readline()

TP=0
total_test=0
choice=1

while line:
    if not line:
        test_tweet=raw_input("Enter Tweet :")
    else:
        line=line.rstrip()
        fields=re.split(r'\t+',line)
        if len(fields)<2:
            line=fp.readline()
            continue
        total_test+=1
    
        print total_test
    
        test_tweet=fields[1]
    
    test_tweet=processTweet(test_tweet,ed,ad,swd)
    test_tweet=" ".join(test_tweet)
    ark_tokenised=tokenizeRawTweetText(test_tweet)

    tweet_size=len(ark_tokenised)

    pos_tfreq=[1 for i in range(tweet_size)]
    neg_tfreq=[1 for i in range(tweet_size)]
    neu_tfreq=[1 for i in range(tweet_size)]

    wn_score=0
    for j in range(tweet_size):
        if ark_tokenised[j] in wn_dict:
                wn_score+=wn_dict[ark_tokenised[j]]
                
    
    for i in range(pos_count):
        for j in range(tweet_size):
            if ark_tokenised[j] in wordict and pos_matrix[i][wordict[ark_tokenised[j]]]==1:
                pos_tfreq[j]+=1
    

    
    #print pos_tfreq
    for i in range(neg_count):
        for j in range(tweet_size):
            if ark_tokenised[j] in wordict and neg_matrix[i][wordict[ark_tokenised[j]]]==1:
                neg_tfreq[j]+=1

    
    
    #print neg_tfreq
    
    for i in range(neu_count):
        for j in range(tweet_size):
            if ark_tokenised[j] in wordict and neu_matrix[i][wordict[ark_tokenised[j]]]==1:
                neu_tfreq[j]+=1

    #print neu_tfreq
    
    pos_uni_prob=10
    for i in range(tweet_size):
        pos_uni_prob*=float(pos_tfreq[j])/float(pos_count+1)


    neg_uni_prob=10
    for i in range(tweet_size):
        neg_uni_prob*=float(neg_tfreq[j])/float(neg_count+1)

    neu_uni_prob=10
    for i in range(tweet_size):
        neu_uni_prob*=float(neu_tfreq[j])/float(neu_count+1)

    pos_given_tweet=pos_prob*pos_uni_prob
    neg_given_tweet=neg_prob*neg_uni_prob
    neu_given_tweet=neu_prob*neu_uni_prob

    print wn_score
    print pos_given_tweet,neg_given_tweet,neu_given_tweet
    if wn_score >0.8:
        pos_given_tweet+=wn_score
    elif wn_score <-0.8:
        neg_given_tweet+=(-1*wn_score)
    print pos_given_tweet,neg_given_tweet,neu_given_tweet
    
    if pos_given_tweet>=neg_given_tweet:
        if pos_given_tweet>=neu_given_tweet:
            print "Classified : positive","Actual : %s" %fields[0]
            if 'positive' in fields[0]:
                TP+=1
        else:
            print "Classified : neutral","Actual : %s" %fields[0]
            if 'neutral'in fields[0]:
                TP+=1
    else:
        if neg_given_tweet>=neu_given_tweet:
            print "Classified : negative","Actual : %s" %fields[0]
            if 'negative' in fields[0]:
                TP+=1
        else:
            print "Classified : neutral","Actual : %s" %fields[0]
            if 'neutral' in fields[0]:
                TP+=1

    line=fp.readline()
    
print "Accuracy : ",float(TP)/float(total_test)
fp.close()
    
