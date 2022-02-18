# train a word2vec model on Amazon corpus
import sys
import gensim

# get path to folder where pos.txt and neg.txt reside
path = sys.argv[1]
# path = "C:\\Users\\xsusa\\repos\\msci-nlp-w22\\assignment-3\\data\\"

# create file handlers for both the Amazon corpus files
pos = open(path + 'pos.txt', 'r')
neg = open(path + 'neg.txt', 'r')
# read lines into a list
pos_lines = pos.read().splitlines()
neg_lines = neg.read().splitlines()

# proprocessing using gensim library
# changing all to lower case
pos_lines = [l.lower() for l in pos_lines]
neg_lines = [l.lower() for l in neg_lines]
# removing stopwords (returns list of strings)
pos_lines = [gensim.parsing.remove_stopwords(l) for l in pos_lines]
neg_lines = [gensim.parsing.remove_stopwords(l) for l in neg_lines]
# tokenize and remove punctuation from corpus (returns list of lists of strings)
pos_tokens = [gensim.utils.simple_preprocess(l) for l in pos_lines]
neg_tokens = [gensim.utils.simple_preprocess(l) for l in neg_lines]
# consolidating all positive and negative reviews
tokens = pos_tokens + neg_tokens

# initialize a gensim model
model = gensim.models.Word2Vec(
    window=2,
    min_count=2,
)

# build vocabulary for the model
model.build_vocab(tokens)

# train model to learn the word embeddings
model.train(tokens, total_examples=model.corpus_count, epochs=model.epochs)

# save model to a file for use in other applications
model.save(path + 'w2v.model')

# close the files
pos.close()
neg.close()