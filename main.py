from src import config
import string
from src.clean_input import load_sentences, clean_sentence
from src.model_similarity import calculate_cosine_similarity_vector_with_sentences_vector,\
    find_most_similar_from_input, print_most_similar_sentence, sentence_to_vec, load_vectors
import numpy as np
import argparse


def main():
    # initialize ArgumentParser class of argparse
    parser = argparse.ArgumentParser()

    # add the argument for the sentences path
    parser.add_argument(
        "--sentences",
        type=str
    )
    # read the arguments from the command line
    args = parser.parse_args()
    path_to_sentences = args.sentences

    # load the sentences from the path given in the argument
    sentences = load_sentences(path_to_sentences)

    # Load embeddings into memory
    print("Loading embeddings")
    embeddings = load_vectors(config.fasttext_model_filename)

    # Ask the user to test sentences as long as he wants
    is_continue = True

    while is_continue:
        # accepts the sentence from the user
        input_sentence = input("Which sentence would you like to test for similarity?\n")

        # Cleaning the sentences
        print("Cleaning the sentences")
        clean_input_sentence = clean_sentence(input_sentence, stop_words=[], punctuation=string.punctuation)
        clean_sentences = [clean_sentence(sentence, stop_words=[], punctuation=string.punctuation)
                           for sentence in sentences]

        # Create sentence embeddings
        input_vec = sentence_to_vec(words=clean_input_sentence.split(), embedding_dict=embeddings)
        input_vec = np.array(input_vec)

        vectors = []
        for sentence in clean_sentences:
            vectors.append(
                sentence_to_vec(
                    words=sentence.split(),
                    embedding_dict=embeddings
                ))

        vectors = np.array(vectors)

        # Builds a dictionary that maps the sentences index (as given in the input) to their cosine similarity
        print("Calculating similarities")
        dict_similarity = calculate_cosine_similarity_vector_with_sentences_vector(input_vector=input_vec,
                                                                                   sentences_vector=vectors)

        # Display the most similar sentence along with the highest similarity score
        most_similar_sentence, most_similar_score = find_most_similar_from_input(sentences, dict_similarity)
        print_most_similar_sentence(input_sentence, most_similar_sentence, most_similar_score)

        # Ask the user if he wants to continue testing
        user_continue = input("Do you want to continue? y/n\n")
        if user_continue == "n":
            break


if __name__ == "__main__":
    main()
