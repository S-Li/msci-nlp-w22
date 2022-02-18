# generate top 20 most similar words for a given word
import sys
import os
import gensim

# get path to .txt file which contains words for evaluation
path = sys.argv[1]
# path = "C:\\Users\\xsusa\\repos\\msci-nlp-w22\\assignment-3\\data\\test.txt"

# open file handler for .txt file
test = open(path, 'r')
# read lines (one word per line) into list
test_words = test.readlines()
# remove new line char from each word
test_words = [word.strip() for word in test_words]
# change all alphabetic characters to lower case
test_words = [word.lower() for word in test_words]

# getting directory to pre-trained W2V model,
# assuming current working directory is ..\assignment-3
path = os.getcwd() + '\\data\\w2v.model'
# load the pre-trained W2V model
model = gensim.models.Word2Vec.load(path)

# generate top 20 most similar words for each test word
for word in test_words:
    try:
        similar_words = model.wv.similar_by_word(word, topn=20)
        print("word: ", word, "\nsimilar words: \n")
        # print words listed in order from most similar to least with index
        for i in range(len(similar_words)):
            print(i + 1, "\t", similar_words[i])
        print("\n")
    # test word not in vocabulary
    except KeyError:
        print("test word ", word, " not in model vocabulary!\n")
        continue

test.close()
