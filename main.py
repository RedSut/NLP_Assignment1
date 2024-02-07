import requests
import nltk
import re

nltk.download('punkt')
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
from autocorrect import Speller


def API_call(page_ids):
    # URL dell'API di Wikipedia
    url = "https://en.wikipedia.org/w/api.php"

    # Parametri della richiesta
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": True,
        "exsectionformat": "plain",
        "redirects": True,
    }

    #titles = ["Medicine", "Python (programming language)"]
    texts = []

    for pd in page_ids:
        params["pageids"] = pd
        # Effettua la richiesta all'API di Wikipedia
        response = requests.get(url, params=params)
        # Estrai il testo della pagina di Wikipedia
        pages = response.json()["query"]["pages"]
        page_id = next(iter(pages))
        page_text = pages[page_id]["extract"]
        texts.append(page_text)
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
    # create the vocabulary
    vocab = set()
    # create the bag-of-words model
    bow_model = []

    i = 0
    for text in texts:
        # create a dictionary to store the word counts
        word_counts = {}
        # tokenize the text
        tokens = nltk.word_tokenize(text)
        # update the vocabulary
        vocab.update(tokens)
        # count the occurrences of each word
        for word in tokens:
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1
        # add the word counts to the bag-of-words model
        bow_model.append(word_counts)

        filename = "bow/" + str(pids[i]) + "_bow.txt"
        with open(filename, 'w', encoding="utf-8") as f:
            f.write(str(word_counts))
        i = i + 1
    return bow_model 


def add_label(texts, label):
    return [(text, label) for text in texts]

def get_classifier(data):
    from nltk.classify import SklearnClassifier
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.feature_selection import SelectKBest, chi2
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([('tfidf', TfidfTransformer()),
                     ('chi2', SelectKBest(chi2, k=1000)),
                     ('nb', MultinomialNB())])
    
    classif = SklearnClassifier(pipeline)

    classif.train(data)

    return classif


if __name__ == "__main__":
    cat_ids = ["18963910", "36248807", "71520442", "14389994", "38262946"] # Geography
    not_cat_ids = ["23862", "18957", "6678", "19009110", "11145"] # Non-Geography

    cat_texts = API_call(cat_ids)
    not_cat_texts = API_call(not_cat_ids)

    cat_pp_texts = preprocessing(cat_texts)
    not_cat_pp_texts = preprocessing(not_cat_texts)

    cat_bow_texts = BOW(cat_pp_texts, cat_ids)
    not_cat_bow_texts = BOW(not_cat_pp_texts, not_cat_ids)

    data = add_label(cat_bow_texts, "Geographic") # Coppia (BOW, Category)
    data += add_label(not_cat_bow_texts, "Non-Geographic") # Coppia (BOW, Non-Category)

    classifier = get_classifier(data)

    new_text = BOW(preprocessing(API_call(["19725260"])), ["19725260"])

    cat = classifier.classify(new_text[0])
    print("The predicted category is: " + cat)