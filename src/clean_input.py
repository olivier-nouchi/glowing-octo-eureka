import json
import sys
import spacy
import string
from nltk.corpus import stopwords

nlp = spacy.load("en_core_web_sm")


def load_sentences(filepath):
    """
    Accepts the filepath from the arguments in the file and gives back a list of the sentences from the json file
    :param filepath: filepath from the CLI, string
    :return: list of sentences, list
    """
    # opening json file with the sentences
    try:
        with open(filepath) as f:
            all_sentences = json.load(f)
            return all_sentences
    # if the file does not exist
    except FileNotFoundError:
        print('File not found. Please make sure that the path is correct')
        sys.exit(1)


def clean_sentence(sentence, stop_words=stopwords.words('english'), punctuation=string.punctuation):
    """
    Clean the sentence, including removing punctuation and stopwords
    :param punctuation: punctuation to remove, list
    :param stop_words: list of stopwords, list
    :param sentence: sentence to clean, string
    :return: cleaned sentence, string
    """
    # "nlp" Object is used to create documents with linguistic annotations.
    doc = nlp(sentence.lower())

    # Tokenize
    words = [token.text for token in doc if token]

    # Remove stopwords
    words = [w for w in words if w not in stop_words]

    # Remove the punctuation from the sentence
    words = [w for w in words if w not in punctuation]

    # Lemmatize the text
    lemmatized_text = " ".join([word.lemma_ for word in nlp(" ".join(words))])

    return lemmatized_text


if __name__ == "__main__":
    pass
