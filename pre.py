import re
import sys
sys.path.insert(0,'ark-tokenizer')
from ark import tokenizeRawTweetText


def processTweet(tweet):
    tweet=tweet.lower()
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','||URL||',tweet)
    tweet = re.sub('@[^\s]+','||AT_USER||',tweet)
    tweet = re.sub('[\s]+', ' ', tweet)
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    tweet = tweet.strip('\'"')
    tweet=tweet.strip(' ')
    return tweet
fp = open(sys.argv[1], 'r')
tokfp = open(sys.argv[2], 'w+')
line = fp.readline()
while line:
    fields=re.split(r'\t+', line)
    polarity=fields[2]
    processedTweet = processTweet(fields[3])
    if(processedTweet != 'not available'):
        tokenizedTweet=tokenizeRawTweetText(processedTweet)
        tokfp.write(polarity+"\t"+" ".join(tokenizedTweet)+"\n")
    line = fp.readline()
#end loop
fp.close()
tokfp.close()
