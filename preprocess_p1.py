import re
import sys
from processTweets_p1 import *
from DictionaryBuilder import *

sys.path.insert(0,'ark-tokenizer')
from ark import tokenizeRawTweetText


fp = open(sys.argv[1], 'r')
tokfp = open(sys.argv[2], 'w+')


line = fp.readline()
while line:
    fields=re.split(r'\t+', line)
    fields[3]=fields[3].strip();
    if(fields[3]!= 'Not Available'):
        polarity=fields[2]
        processedTweet = processTweet(fields[3],ed,ad,swd)
        processedTweet=" ".join(processedTweet)
        tokenizedTweet=tokenizeRawTweetText(processedTweet)
        tokfp.write(polarity+"\t"+" ".join(tokenizedTweet)+"\n")
    line = fp.readline()
#end loop
fp.close()
tokfp.close()
