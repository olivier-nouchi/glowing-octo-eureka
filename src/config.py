import os

# global variables
NUM_FEATURES = 300
MOST_SIMILAR = 0

# name for the models
fasttext_model_filename = "models/crawl-300d-2M.vec"
word2vec_model_filename = "models/GoogleNews-vectors-negative300.bin.gz"

# name for the fasttext model pickle to save
fasttext_model_pickle = os.path.join("pickle", "fasttext_embeddings.pkl")
