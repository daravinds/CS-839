import json
import keras
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import numpy as np
import csv
import string
printable = set(string.printable)

ifile = open('../data/data.csv', "rU", encoding='mac_roman')
reader = csv.reader(ifile, delimiter=",")
print(reader)
training_set = []
i = 0

# Creating training_set
for row in reader:
    # Break after 5000 tweets for now. Change to 1L Later.
    if i > 5000:
        break
    i = i + 1
    training_set.append(row)
print (training_set[0])

train_x = []
for x in training_set:
    train_x.append(str(x[3]))

labels = np.asarray([x[1] for x in training_set])
labels = np.delete(labels, [0])
train_x = np.delete(train_x, [0])

for i in range(len(train_x)):
    train_x[i] = filter(lambda x: x in printable, train_x[i])

# popular words in the tweets
popular_words = 3000

# tokenize the words
tokenizer = Tokenizer(num_words=popular_words)
tokenizer.fit_on_texts(train_x)

# Creating the dictionary
dictionary = tokenizer.word_index

with open('dictionary.json', 'w') as dictionary_file:
    json.dump(dictionary, dictionary_file)

# Converting the words to indices
def words_index(text):
    # Length of every sentence changed to 3k
    return [dictionary[word] for word in kpt.text_to_word_sequence(text)]

# create word indices
word_indices = []

for text in train_x:
    wordIndices = words_index(text)
    word_indices.append(wordIndices)

word_indices = np.asarray(word_indices)

# Converting to a one-hot matrix
train_x = tokenizer.sequences_to_matrix(word_indices, mode='binary')

labels = keras.utils.to_categorical(labels, 2)

# Creating the model. For now we have a sequential model with stack of layers
# One Dense layer with 512 neurons
# Two Dropout layers with sigmoid and softmax
# First layer uses relu for activation

model = Sequential()
model.add(Dense(512, input_shape=(popular_words,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.fit(train_x, labels, batch_size=32, epochs=5, verbose=1,
            validation_split=0.1, shuffle=True)

# Save the model with structure and weights
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

model.save_weights('model.h5')
