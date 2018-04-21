import spacy
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import LabelEncoder


raw_data_path = './data/data.csv'
nlp = spacy.load('en_core_web_md')


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


def load_twitter_gender_data(from_cache=False):
    cached_data_path = raw_data_path + '.cached.pkl'

    if from_cache:
        print('Loading data from cache...')
        with open(cached_data_path, 'rb') as f:
            return pickle.load(f)

    max_length = 1000

    print('Loading and preparing data...')
    raw_data = pd.read_csv(raw_data_path, encoding='latin1', nrows = 1000)

    print('Raw data with 100% confidence:', raw_data.shape)


    # Parse tweet texts
    docs = list(nlp.pipe(raw_data['SentimentText'], batch_size=5000, n_threads=2))

    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(raw_data['Sentiment'])
    y = label_encoder.transform(raw_data['Sentiment'])

    # Pull the raw_data into vectors
    X = extract_features(docs, max_length=max_length)

    # Split into train and test sets
    rs = ShuffleSplit(n_splits=2, random_state=42, test_size=0.2)
    train_indices, test_indices = next(rs.split(X))

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    docs = np.array(docs, dtype=object)
    docs_train = docs[train_indices]
    docs_test = docs[test_indices]

    numeric_data = X_train, y_train, X_test, y_test
    raw_data = docs_train, docs_test, label_encoder

    with open(cached_data_path, 'wb') as f:
        pickle.dump((numeric_data, raw_data), f)

    return numeric_data, raw_data


def load(data_name, *args, **kwargs):
    load_fn_map = {
        'twitter_gender_data': load_twitter_gender_data
    }
    return load_fn_map[data_name](*args, **kwargs)
