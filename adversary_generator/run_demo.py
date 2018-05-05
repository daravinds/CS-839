import csv
import argparse
import pdb

import numpy as np

from datetime import datetime

import model as model_config

from data_utils import load as load_data, extract_features
from adversarial_tools import ForwardGradWrapper, adversarial_paraphrase, \
        _stats_probability_shifts


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
    # Load Twitter gender data
    (_, _, X_test, y_test), (docs_train, docs_test, _) = \
            load_data('twitter_gender_data', from_cache=False)

    # Load model from weights
    model = model_config.build_model()
    model.load_weights(args.model_path)

    # Initialize the class that computes forward derivatives
    grad_guide = ForwardGradWrapper(model)

    # Calculate accuracy on test examples
    preds = model.predict_classes(X_test[:test_samples_cap, ]).squeeze()
    accuracy = np.mean(preds == y_test[:test_samples_cap])
    print('Model accuracy on test:', accuracy)

    # Choose some female tweets
    tweets, = np.where(y_test[:test_samples_cap] == 0)

    print('Crafting adversarial examples...')
    successful_perturbations = 0
    failed_perturbations = 0
    adversarial_text_data = []
    adversarial_preds = np.array(preds)

    for index, doc in enumerate(docs_test[:test_samples_cap]):
        correct_classifed = False
        if y_test[index] == preds[index]:
            adv_doc, (y, adv_y) = adversarial_paraphrase(
                    doc, grad_guide, target=1-y_test[index], use_typos=use_typos, use_synonyms=use_synonyms, mode=mode)
            correct_classifed = True

        if correct_classifed:
            pred = np.round(adv_y)
            if pred != preds[index]:
                successful_perturbations += 1
                print('{}. Successful example crafted.'.format(index))
            else:
                failed_perturbations += 1
                print('{}. Failure.'.format(index))

            adversarial_preds[index] = pred
            adversarial_text_data.append({
                'index': index,
                'doc': clean(doc),
                'adv': clean(adv_doc),
                'success': pred != preds[index],
                'confidence': y,
                'adv_confidence': adv_y
            })
        else:
            print('{}. Misclassified by model - Not generating adversary.'.format(index))
        print('-'*100)
        print('-'*100)

    print('Model accuracy on adversarial examples:',
            np.mean(adversarial_preds == y_test[:test_samples_cap]))
    print('Fooling success rate:',
            successful_perturbations / (successful_perturbations + failed_perturbations))
    print('Average probability shift:', np.mean(
            np.array(_stats_probability_shifts)))

    # Save resulting docs in a CSV file
    with open(args.adversarial_texts_path, 'w') as handle:
        writer = csv.DictWriter(handle,
                fieldnames=adversarial_text_data[0].keys())
        writer.writeheader()
        for item in adversarial_text_data:
            writer.writerow(item)
