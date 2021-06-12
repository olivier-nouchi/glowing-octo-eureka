# Introduction

This script given an input of sentences in a json format and a sentence from the user input will find the most similar sentence from the pool of sentences.

# Results
- If the sentence in input is the same as one of the sentences pool, the result is indeed a similarity of 1.
- "I want us to be together for ever" --> "Inside we both know what's been going on" with a score of 0.84. It looks reasonable.
- "I am eating food in a restaurant" --> "We're no strangers to love" with a score of 0.62. Indeed, the two sentences are not so much related, hence a lower score.

# Installation

1. You can download the needed packages for this project through:

```bash
pip install -r requirements.txt
```

2. Download in the `models` directory the [Fasttext model from this link.](https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip)


This project was developed with python 3.6

# Usage

From the CLI, run the following:

```bash
python3 main.py --sentences input/sentences.json
```

You will be asked to write the sentence you would like to compare to the sentences in input.

You can find the sentences json in the `input` directory.

**Loading of the embeddings may take a few minutes.**

# Further improvements
- Clean further the sentences (gotta --> got to, I've --> I have)
- Add logging
- Add testing functions
- Try other models designed for semantic similarity: Doc2Vec

## License
Open license