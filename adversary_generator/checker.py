import pandas as pd

types = []
raw_data = pd.read_csv('./data/finalcleantweets.csv', encoding='latin1')

# for i, tweet in enumerate(raw_data.tweet):
#      if type(tweet)==str:
#          types.append(1)
#      else:
#          types.append(0)
#
# raw_data['finaltypes'] = types
# # print (raw_data['types'])
# #
# raw_data = raw_data[raw_data.finaltypes != 0]
# #
for tweet in raw_data['tweet']:
    if(type(tweet) != str):
        print(type(tweet))
        print (tweet)
#raw_data.to_csv("./data/finalcleantweets.csv")
