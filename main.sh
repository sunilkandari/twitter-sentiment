#!/bin/bash

python preprocess.py traintest.tsv preprocessedTweets.txt
echo "Preprocessing successfull !! (Saved to preprocessedTweets.txt and preprocessedTweets_vs_Feature.txt)"
#python preprocess.py testing.tsv testpreprocessedTweets.txt
#echo "Preprocessing successfull !! (Saved to testpreprocessedTweets.txt)"
if [ $# -lt 2 ]; then
  python unigramSVM.py $1
else
  python unigramSVM.py $1 $2
fi
