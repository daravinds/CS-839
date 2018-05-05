import csv
import argparse
import pdb
import spacy

import numpy as np

from datetime import datetime

import model as model_config
nlp = spacy.load('en_core_web_md')
#from data_utils import load as load_data, extract_features
#from adversarial_tools import ForwardGradWrapper, adversarial_paraphrase, \
#        _stats_probability_shifts
def extract_features(docs, max_length):
    docs = list(docs)
    X = np.zeros((len(docs), max_length), dtype='int32')
    for i, doc in enumerate(docs):
        j = 0
        for token in doc:
            if token.has_vector and not token.is_punct and not token.is_space:
                X[i, j] = token.rank + 1
                j += 1
                if j >= max_length:
                    break
    return X


parser = argparse.ArgumentParser(
        description='Craft adversarial examples for a text classifier.')
parser.add_argument('--model_path',
                    help='Path to model weights',
                    default='./data/model.dat')
parser.add_argument('--adversarial_texts_path',
                    help='Path where results will be saved',
                    default='./data/adversarial_texts.csv')
parser.add_argument('--test_samples_cap',
                    help='Amount of test samples to use',
                    type=int, default=10)
parser.add_argument('--mode',
                    help='Mode of running --- ' +
                          '1. Only insertion (or) removal' +
                          '2. Only synonym (or) typographic replacement' +
                          '3. Synonym (and) insertion (or) removal' +
                          '4. Greedy combination of 1,2 and 3',
                    type=int, default=3)


def clean(text):
    '''
    Clean non-unicode characters
    '''
    return ''.join([i if ord(i) < 128 else ' ' for i in str(text)])


if __name__ == '__main__':
    args = parser.parse_args()
    test_samples_cap = args.test_samples_cap
    mode = args.mode

    print('Mode is ', mode)

    use_typos = False
    use_synonyms = False

    if mode == 1:
        use_typos = False
        use_synonyms = False
    elif mode == 2:
        use_typos = True
        use_synonyms = True
    elif mode == 3:
        use_typos = False
        use_synonyms = True
    elif mode == 4:
        use_typos = True
        use_synonyms = True
    else:
        print('Enter mode to execute. Quitting....')
        exit()


    print('Running mode and use_typos ', args.mode, use_typos)

    # Load CSV DATA HERE
    max_length = 1000
    raw_data = ["Hello I am good", "He is bad"]
    docs = list(nlp.pipe(raw_data, batch_size=5000, n_threads=2))
    X = extract_features(docs, max_length=max_length)

    # Load model from weights
    model = model_config.build_model()
    model.load_weights(args.model_path)

    # Get prediction
    pred = model.predict_classes(X).squeeze()
    docs = np.array(docs, dtype=object)
    print (docs)

    for doc in enumerate(docs):
        print (doc[1])
        x = extract_features([doc[1]], max_length=max_length)[0]
        y = model.predict(x.reshape(1, -1), verbose=0).squeeze()
        print ("confidence", y)

    print("predictions", pred)

    # WRITE BACK TO CSV HERE
    # # Save resulting docs in a CSV file
    # with open(args.adversarial_texts_path, 'w') as handle:
    #     writer = csv.DictWriter(handle,
    #             fieldnames=adversarial_text_data[0].keys())
    #     writer.writeheader()
    #     for item in adversarial_text_data:
    #         writer.writerow(item)
