from collections import defaultdict

specialChar='1234567890#@%^&()_=`{}:"|[]\;\',./\n\t\r '

#Create wordnet dictionary
def getWordnetDictionary():
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
    return wn_dict


def getEmoticonDictionary():
    f=open("emoticonsWithPolarity.txt",'r')
    data=f.read().split('\n')
    emoticonsDict={}
    for i in data:
        if i:
            i=i.split()
            value=i[-1]
            key=i[:-1]
            for j in key:
                emoticonsDict[j]=value
    f.close()
    return emoticonsDict

def getAcronymDictionary():
    f=open("acronym.txt",'r')
    data=f.read().split('\n')
    acronymDict={}
    for i in data:
        if i:
            i=i.split('\t')
            key=i[0]
            value=i[1]
            acronymDict[key]=value
    f.close()
    return acronymDict

def getStopwordDictionary():
    stopWords=defaultdict(int)
    f=open("stopWords.txt", "r")
    for line in f:
        if line:
            line=line.strip(specialChar).lower()
            stopWords[line]=1
    f.close()
    return stopWords
