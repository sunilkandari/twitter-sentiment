import re
import nltk
from DictionaryBuilder import *


#Creating Dictionaries
ed=getEmoticonDictionary()
ad=getAcronymDictionary()
swd=getStopwordDictionary()

def getPOSScore(tweet):
    count = len(tweet)
    listp = nltk.pos_tag(tweet)
    a=0
    nouns_cnt = 0
    prep_cnt = 0
    adj_cnt = 0
    while a < count:
        if(listp[a][1] == 'NN'):
            nouns_cnt = nouns_cnt + 1
        elif(listp[a][1] == 'IN'):
            prep_cnt = prep_cnt + 1
        elif(listp[a][1] == 'JJ'):
            adj_cnt = adj_cnt + 1
        a = a+1
    return nouns_cnt,prep_cnt,adj_cnt

specialChar='1234567890#@%^&()_=`{}:"|[]\;\',./\n\t\r '

""" eg greaaaaaaaaaaaaat->greaaat
        param: tweet - list of words in tweet
        return: list of words and count of words which has repitetion"""
def replaceRepetition(tweet):
    count=0
    for i in range(len(tweet)):
        x=list(tweet[i])
        if len(x)>3:
            flag=0
            for j in range(3,len(x)):
                if(x[j-3]==x[j-2]==x[j-1]==x[j]):
                    x[j-3]=''
                    if flag==0:
                        count+=1
                        flag=1
            tweet[i]=''.join(x).strip(specialChar)
    return tweet,count

"""remove the non-english or better non-ascii characters
        param: list of words in tweets
        return: tweet with English words only and the count of words removed."""
def removeNonEnglishWords(tweet):
    newTweet=[]
    count=0
    for i in range(len(tweet)):
        if tweet[i]!='':
            chk=re.match(r'([a-zA-z0-9 \+\?\.\*\^\$\(\)\[\]\{\}\|\\/:;\'\"><,.#@!~`%&-_=])+$',tweet[i])
            if chk:
                count+=1
                newTweet.append(tweet[i])
    return newTweet,count

"""Removes the stopwords.
    param: list of words in tweet,a Dictonary of stopword.
    return: modified list words """
def removeStopWords(stopWordsDict,tweet):        
    newTweet=[]
    for i in range(len(tweet)):
        if tweet[i].strip(specialChar) not in stopWordsDict:
            newTweet.append(tweet[i])
    return newTweet


""" replaces the emoticons present in tweet with its polarity
    param : emoticons dictioary emoticons as key polarity as value
    return: list which contains words in tweet and return list of words in tweet after replacement"""

def replaceEmoticons(emoticonsDict,tweet):
    isEmoticonPresent=0
    for i in range(len(tweet)):
        if tweet[i] in emoticonsDict:
            isEmoticonPresent=1
            tweet[i]=emoticonsDict[tweet[i]]
    return tweet,isEmoticonPresent


"""expand the Acronym in tweet
    param: acronym dictionary ,acronym as key and abbreviation as value,list of words in tweet.
    return: list of words in tweet after expansion and their count"""
def expandAcronym(acronymDict,tweet):
    count=0
    newTweet=[]
    for i in range(len(tweet)):
        word=tweet[i].strip(specialChar)
        if word:
            if word in acronymDict:
                count+=1
                newTweet+=acronymDict[word].split(" ")
            else:
                newTweet+=[tweet[i]]
    return newTweet,count


"""param: list of words in tweet
   return: list of words in tweet after expanding"
       eg isn't -> is not """
def expandNegation(tweet):
    newTweet=[]
    for i in range(len(tweet)):
        word=tweet[i].strip(specialChar)
        if(word[-3:]=="n't"):
            if word[-5:]=="can't" :
                newTweet.append('can')
            else:
                newTweet.append(word[:-3])
            newTweet.append('not')
        else:
            newTweet.append(tweet[i])
    return newTweet

"""param: a list which contains words in tweet.
   return: list of words in tweet after replacement ("not","n't","no","~")
       eg.
       not -> negation
       isn't -> negation """
def replaceNegation(tweet):
    for i in range(len(tweet)):
        word=tweet[i].lower().strip(specialChar)
        if(word=="no" or word=="not" or word.count("n't")>0):
            tweet[i]='negation'
    return tweet
"""
replace url with IURLI
"""
def replaceURL(tweet):
    tweet=re.sub('((www\.[^\s]+)|(https?://[^\s]+))','IURLI',tweet)
    return tweet

"""
eg: replace @sunil with IATUSERI
"""
def replaceTarget(tweet):
    tweet=re.sub('@[^\s]+','IATUSERI',tweet)
    return tweet

""" param: tweet as a string
    return: list of words in tweet after removing numbers """
def removeNumbers(tweet):
    tweet=re.sub('^[0-9]+', '', tweet)
    return tweet

"""param: string tweet
       return: list of words in tweet after replacement 
       eg : #*** - > *** """
def replaceHashtag(tweet):
    tweet=re.sub(r'#([^\s]+)', r'\1', tweet)
    return tweet

def mergeSpace(tweet):
    return re.sub('[\s]+', ' ', tweet)

""" Intial preprocessing
    param: tweet string
    return: preprocessed tweet """
def processTweet(tweet,ed,ad,swd):
    #Other Feature List (NON_ENG,REPEAT,EMOTICON,ACRONYM and WN_SCORE)
    features=[]
    
    tweet=tweet.lower()
    tweet = replaceURL(tweet)
    tweet = replaceTarget(tweet)
    tweet = replaceHashtag(tweet)
    #print "After url hashtag target",tweet
    
    tweet = mergeSpace(tweet)
    tweet = tweet.strip('\'"')
    tweet=tweet.strip(' ')
    tweet=tweet.split(" ")
	   
    tweet,count=removeNonEnglishWords(tweet)
    
    features.append(str(count))
    #print "Non English",tweet
    
    tweet,count=replaceRepetition(tweet)
    features.append(str(count))
    
    #print "Repetition",tweet

    tweet,count=replaceEmoticons(ed,tweet)
    features.append(str(count))
    
    #print "Emoticons",tweet
    
    tweet,count=expandAcronym(ad,tweet)
    features.append(str(count))
    
    #print "Acronym",tweet
    tweet=expandNegation(tweet)
    tweet=replaceNegation(tweet)
    #print "Negation",tweet
    tweet=removeStopWords(swd,tweet)
    for i in xrange(len(tweet)-1,-1,-1):
        if tweet[i] == '':
            tweet.pop(i)
    n,p,a=getPOSScore(tweet)
    features.append(str(n))
    features.append(str(p))
    features.append(str(a))

    #print "Stop Words",tweet
    return tweet,features

