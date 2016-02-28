import sys
import json

def main():
    allTerms = {}
    total = 0

    tweetFile = open(sys.argv[1])

    for line in tweetFile:
        tweet = json.loads(line)

        if 'text' in tweet:
            for word in tweet["text"].encode('utf-8').split():
                total += 1
                if word in allTerms:
                    allTerms[word] += 1
                else:
                    allTerms[word] = 1

    for term in allTerms:
        print str(term) + " " + str((float) (allTerms[term]) / total)

if __name__ == '__main__':
    main()
