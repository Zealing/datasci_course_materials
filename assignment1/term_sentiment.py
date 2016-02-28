import sys
import json

sentimenScores = {}

def afinnSentimentBuildup():
    afinnFile = open(sys.argv[1])
    for line in afinnFile:
        term, score = line.split("\t")
        sentimenScores[term] = int(score)

def rateTweets():
    tweetFile = open(sys.argv[2])
    newSentimentScores = {}

    for line in tweetFile:
        tweet = json.loads(line)
        sentimenScore = 0;
        if 'text' in tweet:
            tweetWords = tweet["text"].encode('utf-8').split()

            # calculate the total score of this tweet
            for word in tweetWords:
                if word in sentimenScores:
                    sentimenScore += sentimenScores[word]

            # for word not in sentiment list, calculate its score based on that tweet
            for word in tweetWords:
                if word not in sentimenScores:
                    if word not in newSentimentScores:
                        newSentimentScores[word] = float(sentimenScore) / len(tweetWords)
                    else:
                        newSentimentScores[word] += float(sentimenScore) / len(tweetWords)

    for newSentimentWord, newSentimentWordScore in newSentimentScores.items():
        print str(newSentimentWord) + " " + str(newSentimentWordScore)

def main():
    afinnSentimentBuildup()
    rateTweets()

if __name__ == '__main__':
    main()
