from twitterscraper import query_tweets

# All tweets matching bitcoint will be returned.
# You will get at least 10 results within the minimal
# possible time/number of requests
# For now we are using an other dataset.
# In the ML project, we will shift to bitcoint tweets

i = 1
for tweet in query_tweets("#bitcoin", 1000)[:1000]:
    print i
    print(tweet.text.encode('utf-8'))
    i += 1
