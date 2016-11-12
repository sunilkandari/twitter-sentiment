import re
import sys
from processTweets import *
from DictionaryBuilder import *

sys.path.insert(0,'ark-tokenizer')
from ark import tokenizeRawTweetText


fp = open(sys.argv[1], 'r')
tokfp = open(sys.argv[2], 'w+')

featurefp=open("processedTweet_vs_features.txt",'w+')

wn_dict=getWordnetDictionary()

line = fp.readline()
while line:
    fields=re.split(r'\t+', line)
    fields[3]=fields[3].strip();
    if(fields[3]!= 'Not Available'):
        polarity=fields[2]
        processedTweet,featureVect = processTweet(fields[3],ed,ad,swd)
        processedTweet=" ".join(processedTweet)
        
        tokenizedTweet=tokenizeRawTweetText(processedTweet)

        tweet_size=len(tokenizedTweet)
        #Find word polarity score of the word using word net
        
        wn_score=0
        for j in range(tweet_size):
            if tokenizedTweet[j] in wn_dict:
                wn_score+=wn_dict[tokenizedTweet[j]]
        featureVect.append(str(wn_score))

        #Write coresp. processed tweet and feature vector 2 corresp. file
        tokfp.write(polarity+"\t"+" ".join(tokenizedTweet)+"\n")
        featurefp.write(" ".join(featureVect)+"\n")
    line = fp.readline()
#end loop
fp.close()
tokfp.close()
