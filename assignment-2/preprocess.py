# preprocess.py uses NLTK to tokenize the Amazon reviews corpus, 
# labels each review, and splits them into sets 
# (80% training, 10% validation, 10% test)

import sys
import nltk
from nltk import tokenize as tok
from nltk.corpus import stopwords
import random

# process command line argument for path to split data sets
path = sys.argv[1] # skip first argument which is script name
# path = 'C:\\Users\\xsusa\\repos\\msci-nlp-w22\\assignment-2\\data\\'

# download nltk data for tokenization and stopword removal
nltk.download('punkt')
nltk.download('stopwords')

# define special characters to be removed from tokenized sentences
specialChars = """.,-!?''"#$%&()*+/:;<=>@[\\]^``{|}~\t\n"""

# open the data files
pos = open(path + 'pos.txt', 'r')
neg = open(path + 'neg.txt', 'r')
# posSample = open(path + 'pos_sample.txt', 'r')
# negSample = open(path + 'neg_sample.txt', 'r')
stopWords = set(stopwords.words('english'))

# create and open output files
out = open(path + 'out.csv', 'w', newline="")
train = open(path + 'train.csv', 'w', newline="")
val = open(path + 'val.csv', 'w', newline="")
test = open(path + 'test.csv', 'w', newline="")
outNoStop = open(path + 'out_ns.csv', 'w', newline="")
trainNoStop = open(path + 'train_ns.csv', 'w', newline="")
valNoStop = open(path + 'val_ns.csv', 'w', newline="")
testNoStop = open(path + 'test_ns.csv', 'w', newline="")
# output files for labels
trainLabels = open(path + 'train_labels.csv', 'w', newline="")
valLabels = open(path + 'val_labels.csv', 'w', newline="")
testLabels = open(path + 'test_labels.csv', 'w', newline="")

# remove the special characters !"#$%&()*+/:;<=>@[\\]^`{|}~\t\n 
# input: tokenized sentence.
# output: tokenized sentence with special character tokens removed.
def removeSpecialChars(tokens):
    filteredTokens = [t for t in tokens if not t in specialChars]
    return filteredTokens

# remove stopwords from the NLTK list of english stopwords.
# input: list of tokens.
# output: list of tokens with stop words removed.
def removeStopWords(tokens):
    filteredTokens = [t for t in tokens if not t.lower() in stopWords]
    return filteredTokens 

# randomly split the dataset into 80% training, 10% validation, 10% testing.
# input: none.
# output: tuple of (writer, writer ns, writer labels) for set the token belongs to.
def chooseDataSet():
    number = random.randrange(0, 99) # default step size 1
    if 20 <= number <= 29:
        return((val, valNoStop, valLabels))
    elif 50 <= number <= 59:
        return((test, testNoStop, testLabels))
    else:
        return((train, trainNoStop, trainLabels))

posLines = pos.readlines()
negLines = neg.readlines()
numLines = len(posLines) # both pos and neg files same length

# iterate through corpus and output into out files after 
# tokenization, punctuation filtering, and stopword removal.
# alternating between pos and neg for out.csv and out_ns.csv
# randomly assigning different data sets (tr, val, tst) for pos and neg.
for index in range(numLines): 
    # get both positive and negative sentences
    posLine = posLines[index]
    negLine = negLines[index]

    # tokenize the sentence including punctuation marks as tokens
    posLineTok = tok.word_tokenize(posLine)
    negLineTok = tok.word_tokenize(negLine)
    
    # remove punctuation marks as independent tokens
    posLineTok = removeSpecialChars(posLineTok)
    negLineTok = removeSpecialChars(negLineTok)

    # write tokenized sentences to out.csv
    out.write(str(posLineTok) + '\n')
    out.write(str(negLineTok) + '\n') 

    # randomly choose dataset for positive sentence (train, val, test)
    writersRandom = chooseDataSet() # returns tuple (writer, writerNs)
    writersRandom[0].write(str(posLineTok) + '\n') # write to selected file with stopwords

    posLineTokNs = removeStopWords(posLineTok) # remove stopwords from sentence
    outNoStop.write(str(posLineTokNs) + '\n') # write tokenized sentence without stopwords to out_ns.csv
    writersRandom[1].write(str(posLineTokNs) + '\n') # write to selected file without stopwords
    writersRandom[2].write('1\n') # write positive label to labels file

    # randomly choose dataset for negative sentence (train, val, test)
    writersRandom = chooseDataSet() # returns tuple (writer, writerNs)
    writersRandom[0].write(str(negLineTok) + '\n') # write to selected file with stopwords
    
    negLineTokNs = removeStopWords(negLineTok) # remove stopwords from sentence
    outNoStop.write(str(negLineTokNs) + '\n') # write tokenized sentence without stopwords to out_ns.csv
    writersRandom[1].write(str(negLineTokNs) + '\n') # write to selected file without stopwords
    writersRandom[2].write('0\n') # write negative label to labels file

# close all files
pos.close()
neg.close()
# negSample.close()
# posSample.close()
out.close()
train.close()
val.close()
test.close()
outNoStop.close()
trainNoStop.close()
valNoStop.close()
testNoStop.close()
trainLabels.close()
valLabels.close()
testLabels.close()