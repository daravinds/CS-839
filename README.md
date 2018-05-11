# CS-839

This is an adversarial ML project where we create adversarial examples
for tweets.

We have built a sentiment analyzer based on a DNN. The DNN has one hidden layer
and two dropout layers. It performed with about 80% accuracy on the set-aside
dataset.

The adversaries were generated using two methods based on references we read -
1) Synonyms for words which contribute most to the label
2) Insertion/Deletion/Modification of important words
