import requests
import nltk
import re

nltk.download('punkt')
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
from autocorrect import Speller

save_texts = False

def API_call(page_ids):
    # Wikipedia API URL
    url = "https://en.wikipedia.org/w/api.php"

    # Request parameters
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": True,
        "exsectionformat": "plain",
        "redirects": True,
    }

    texts = []

    for pd in page_ids:
        params["pageids"] = pd
        # Wikipedia API request
        response = requests.get(url, params=params)
        # Page text extraction
        pages = response.json()["query"]["pages"]
        page_id = next(iter(pages))
        page_text = pages[page_id]["extract"]
        texts.append(page_text)
        
        # Save texts on files
        if save_texts:
            filename = "texts/" + str(page_id) + "_text.txt"
            with open(filename, 'w', encoding="utf-8") as f:
                f.write(page_text)
        
    return texts


def preprocessing(texts):
    pp_texts = []
    for t in texts:
        # Remove none alphabetic characters
        sms = re.sub('[^A-Za-z]', ' ', t)
        # Make the word lower case
        sms = sms.lower()
        # Remove the stop words
        tokenized_sms = word_tokenize(sms)
        for word in tokenized_sms:
            if word in stopwords.words('english'):
                tokenized_sms.remove(word)
        # Stemming
        stemmer = PorterStemmer()
        for i in range(len(tokenized_sms)):
            tokenized_sms[i] = stemmer.stem(tokenized_sms[i])
        # Spell correction
        spell = Speller(lang='en')
        tokenized_sms[i] = stemmer.stem(spell(tokenized_sms[i]))
        
        sms_text = " ".join(tokenized_sms)
        pp_texts.append(sms_text)
    
    return pp_texts


def BOW(texts, pids):
    # Create the vocabulary
    vocab = set()
    # Create the bag-of-words model
    bow_model = []

    i = 0
    for text in texts:
        # Create a dictionary to store the word counts
        word_counts = {}
        # Tokenize the text
        tokens = nltk.word_tokenize(text)
        # Update the vocabulary
        vocab.update(tokens)
        # Count the occurrences of each word
        for word in tokens:
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1
        # Add the word counts to the bag-of-words model
        bow_model.append(word_counts)

        # Save bag-of-words models on files
        if save_texts:
            filename = "bow/" + str(pids[i]) + "_bow.txt"
            with open(filename, 'w', encoding="utf-8") as f:
                f.write(str(word_counts))
            i = i + 1
    return bow_model 


def add_label(texts, label):
    # Add a the same label to multiple texts
    return [(text, label) for text in texts]

def get_classifier(data):
    from nltk.classify import SklearnClassifier
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.feature_selection import SelectKBest, chi2
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline

    # Create the pipeline
    pipeline = Pipeline([('tfidf', TfidfTransformer()),
                     ('chi2', SelectKBest(chi2, k=1000)),
                     ('nb', MultinomialNB())])
    
    # Create the classifier
    classif = SklearnClassifier(pipeline)
    # Train dataset
    classif.train(data)

    return classif


if __name__ == "__main__":
    # Texts ids
    cat_ids = ["18963910", "36248807", "71520442", "14389994", "38262946"] # Geographic
    not_cat_ids = ["23862", "18957", "6678", "19009110", "11145"] # Non-Geographic

    # Get texts from Wikipedia API
    cat_texts = API_call(cat_ids)
    not_cat_texts = API_call(not_cat_ids)

    # Preprocessing
    cat_pp_texts = preprocessing(cat_texts)
    not_cat_pp_texts = preprocessing(not_cat_texts)

    # Create BOW representation
    cat_bow_texts = BOW(cat_pp_texts, cat_ids)
    not_cat_bow_texts = BOW(not_cat_pp_texts, not_cat_ids)

    # Create the training dataset
    train_data = add_label(cat_bow_texts, "Geographic") # Coppia (BOW, Category)
    train_data += add_label(not_cat_bow_texts, "Non-Geographic") # Coppia (BOW, Non-Category)

    # Get the classifier
    classifier = get_classifier(train_data)

    #new_text = BOW(preprocessing(API_call(["19725260"])), ["19725260"])

    # Classify the new text
    with open("input_text.txt", "r", encoding="utf-8") as f:
        new_text = f.read()
    new_bow_text = BOW(preprocessing([new_text]), ["new"])        

    cat = classifier.classify(new_bow_text[0])
    print("The predicted category is: " + cat)