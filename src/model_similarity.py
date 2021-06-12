import sys

import numpy as np
from src import config
import io
import pickle


def sentence_to_vec(words, embedding_dict):
    """
    Given a sentence and other information, this function returns embedding for the whole sentence
    :param words: sentence, string
    :param embedding_dict: dictionary word:vector
    :return: normalized vector of the sentence
    """

    # initialize empty list to store embeddings
    M = []
    for w in words:
        # for every word, fetch the embedding from the dictionary and append to the list of embeddings
        if w in embedding_dict:
            M.append(embedding_dict[w])

    # if we don't have any vectors returns zeros
    if not M:
        return np.zeros(config.NUM_FEATURES)

    # convert list of embeddings to array
    M = np.array(M)

    # calculate sum over axis=0
    v = M.sum(axis=0)

    # return normalized vector
    return v / np.sqrt((v ** 2).sum())


def load_vectors(fname):
    """
    Loads vectors either from pickle file if mapping already exists, otherwise parse the file form fasttext model
    :param fname: filename of the fasttext model, string
    :return: dictionary word:vector
    """

    try:
        # try loading the embeddings mapping from the pickle already
        with open(config.fasttext_model_pickle, 'rb') as handle:
            data = pickle.load(handle)
        print("Loaded model from pickle file")

    except FileNotFoundError:
        try:
            fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            # beginning of the file is the number of vectors and dimensions
            num, dim = map(int, fin.readline().split())
            data = {}
            for line in fin:
                tokens = line.split(' ')
                data[tokens[0]] = list(map(float, tokens[1:]))

            # save the model in a pickle
            print("Saving the model in a pickle file")
            with open(config.fasttext_model_pickle, 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        except FileNotFoundError:
            print("Model file not found.")
            sys.exit(1)

    return data


def calculate_cosine_similarity_vector_with_sentences_vector(input_vector, sentences_vector):
    """
    Calculate the cosine similarity between an input vector and sentences vector
    :param input_vector: the embedded vector from the input sentence, numpy array
    :param sentences_vector: the embedded sentences vector, numpy array
    :return: dictionary sentence:score
    """
    dictionary_cosine_similarity = dict()

    for index_vector, sentence_vector in enumerate(sentences_vector):
        vectors_product = np.dot(sentence_vector, input_vector)
        norms_product = np.linalg.norm(sentence_vector) * np.linalg.norm(input_vector)
        cosine_similarity = vectors_product / norms_product

        dictionary_cosine_similarity[index_vector] = cosine_similarity

    return dictionary_cosine_similarity


def find_most_similar_from_input(sentences, dictionary_similarity):
    """
    Find the highest score in the mapping sentence:score
    :param sentences: sentences
    :param dictionary_similarity: mapping index:score
    :return: most similar sentence and score, tuple
    """
    most_similar = sorted(dictionary_similarity.items(), key=lambda x: x[1], reverse=True)[config.MOST_SIMILAR]
    sentence_index, most_similar_score = most_similar

    return sentences[sentence_index], most_similar_score


def print_most_similar_sentence(input_sentence, most_similar_sentence, most_similar_score):
    """
    Format the output of the script to show the most similar sentence with the highest similarity score
    :param input_sentence: sentence from the user, string
    :param most_similar_sentence: most similar sentence (cosine similarity based), string
    :param most_similar_score: highest similarity score, float
    :return: None
    """
    print(f"The input sentence is {input_sentence}")
    print(f"The most similar sentence is: '{most_similar_sentence}'\n"
          f"Highest similarity score of: {round(most_similar_score,2)}")


if __name__ == "__main__":
    pass
