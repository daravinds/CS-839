import json
import numpy as np
import keras
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
from keras.models import model_from_json

# tokenizer to set the tweet up in one-hot manner
tokenizer = Tokenizer(num_words=3000)

# Either positive or negative label
labels = ['negative', 'positive']

# dictionary with the word and indices
with open('dictionary.json', 'r') as dictionary_file:
    dictionary = json.load(dictionary_file)

# Getting indices
def words_index(text):
    words = kpt.text_to_word_sequence(text)
    wordIndices = []
    for word in words:
        if word in dictionary:
            wordIndices.append(dictionary[word])
        else:
            print("'%s' Not in dictionary." %(word))
    return wordIndices

# Read the nnet structure
nnet_file = open('model.json', 'r')
nnet_structure = nnet_file.read()
nnet_file.close()

# Create a model from that
model = model_from_json(nnet_structure)

# Save weights your nodes with your saved values
model.load_weights('model.h5')

# Fun part where we can test the network
while 1:
    sentence = raw_input('Input a sentence to be evaluated, or Enter to quit: ')

    if len(sentence) == 0:
        break

    # format your input for the neural net
    test_word_index_format = words_index(sentence)
    input = tokenizer.sequences_to_matrix([test_word_index_format], mode='binary')

    # predict which bucket your input belongs in
    prediction = model.predict(input)

    # print the result
    print("%s sentiment; %f%% confidence" % (labels[np.argmax(pred)],
           prediction[0][np.argmax(prediction)] * 100))
