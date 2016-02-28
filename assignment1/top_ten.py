import sys
import json
import operator
import math

hashtags = {}

def main():
	tweetFile = open(sys.argv[1])

	for line in tweetFile:
		tweet = json.loads(line)

		# first store the hashtags in dict
		if 'entities' in tweet and len(tweet['entities']['hashtags']) != 0:
			tagsContent = tweet['entities']['hashtags'][0]['text'].encode('utf-8')
			if tagsContent is not None:
				if hashtags.has_key(tagsContent):
					hashtags[tagsContent] += 1
				else:
					hashtags[tagsContent] = 1
			else:
				continue

	# then sort the dict to find out the top 10
	topTags = sorted(hashtags.items(), key = operator.itemgetter(1), reverse = True)[:10]
	for item in topTags:
		print item[0] + " " + str(item[1])

if __name__ == '__main__':
	main()