# NLP Assignment 1
# Author: Davide Sut 
# ID: VR505441

## The Assignment

Five different Wikipedia pages were chosen for each category: Geographic, Non-Geographic.
The pages texts were extracted using the Wikipedia API and can be stored into a .txt file.

The texts are preprocessed using in order:
- A regular expression to remove all non alphabetic characters;
- A function to transform each word in lowercase;
- Tokenization using nltk word_tokenize;
- Removing stopwords using nltk english stopwords corpus;
- Stemming using Porter's algorithm;
- Spell correction using autocorrect library.

After the preprocessing step, the bag-of-words is created for each text and can be stored into a .txt file.
Finally, the labels are applied to each BoW and a training set is created.

The classifier is built and trained using the nltk NaiveBayesClassifier.

## Instructions

You simply need to put the text to classify into _input\_text.txt_ file.

Then run the _main.py_ script and the console tells you the predicted class for that text.
