import sys
import re
import csv
import random

# process command line argument for path to neg.txt and pos.txt
path = sys.argv[1] # skip first argument which is script name
# path = 'C:\\Users\\xsusa\\repos\\msci-nlp-w22\\assignment-1\\data\\'

# open the data files
neg = open(path + 'neg.txt', 'r')
pos = open(path + 'pos.txt', 'r')
# negSample = open(path + 'neg_sample.txt', 'r')
# posSample = open(path + 'pos_sample.txt', 'r')
sw = open(path + 'NLTK_list_of_english_stopwords.txt', 'r')

# create and open output files
out = open(path + 'out.csv', 'w', newline="")
train = open(path + 'train.csv', 'w', newline="")
val = open(path + 'val.csv', 'w', newline="")
test = open(path + 'test.csv', 'w', newline="")
outNoStop = open(path + 'out_ns.csv', 'w', newline="")
trainNoStop = open(path + 'train_ns.csv', 'w', newline="")
valNoStop = open(path + 'val_ns.csv', 'w', newline="")
testNoStop = open(path + 'test_ns.csv', 'w', newline="")

# creating csv writer objects for each output file
writerOut = csv.writer(out)
writerTrain = csv.writer(train)
writerVal = csv.writer(val)
writerTest = csv.writer(test)
writerOutNs = csv.writer(outNoStop)
writerTrainNs = csv.writer(trainNoStop)
writerValNs = csv.writer(valNoStop)
writerTestNs = csv.writer(testNoStop)

# collect all the stop words into a list
stopWords = []
for word in sw.readlines():
    stopWords.append(word[:-1])

# remove the special characters !"#$%&()*+/:;<=>@[\\]^`{|}~\t\n 
# and all other punctuation marks from each raw string.
# input: raw string.
# output: string with special characters removed.
def removeSpecialChars(rawStr):
    rawStr = re.sub(r'[^\w\s]', '', rawStr) # \w = [a-zA-Z0-9_], \s = whitespace
    rawStr = re.sub(' +', ' ', rawStr) # remove extra spaces
    return rawStr

# remove stopwords from the NLTK list of english stopwords.
# input: list of tokens.
# output: list of tokens with stop words removed.
def removeStopWords(tokens):
    for stopWord in stopWords:
        tokens = [word for word in tokens if word != stopWord]
        # eliminating any stop words with capitalizations
        tokens = [word for word in tokens if word != stopWord.capitalize()]
    return tokens 

# randomly split the dataset into 80% training, 10% validation, 10% testing.
# input: none.
# output: tuple of (writer, writer ns) for set the token belongs to.
def chooseDataSet():
    number = random.randrange(0, 99) # default step size 1
    if 20 <= number <= 29:
        return((writerVal, writerValNs))
    elif 50 <= number <= 59:
        return((writerTest, writerTestNs))
    else:
        return((writerTrain, writerTrainNs))

# count = 0 # temp var
# limit = 500 # temp var

negLines = neg.readlines()
posLines = pos.readlines()
numLines = len(negLines) # both pos and neg files same length

# iterate through corpus and output into out files after 
# tokenization, punctuation filtering, and stopword removal.
# alternating between pos and neg for out.csv and out_ns.csv
# randomly assigning different data sets (tr, val, tst) for pos and neg.
for index in range(numLines): 
    # get both positive and negative sentences
    negLine = negLines[index]
    posLine = posLines[index]

    # remove EOL character
    negLine = negLine.strip() 
    posLine = posLine.strip()

    # remove all punctuation marks
    negLine = removeSpecialChars(negLine)
    posLine = removeSpecialChars(posLine)

    # tokenize the sentence
    negLineTok = re.split(" ", negLine) 
    posLineTok = re.split(" ", posLine) 

    # write tokenized sentences to out.csv
    writerOut.writerow(negLineTok)
    writerOut.writerow(posLineTok) 

    # randomly choose dataset for negative sentence (train, val, test)
    writersRandom = chooseDataSet() # returns tuple (writer, writerNs)
    writersRandom[0].writerow(negLineTok) # write to selected file with stopwords

    negLineTokNs = removeStopWords(negLineTok) # remove stopwords from sentence
    writerOutNs.writerow(negLineTokNs) # write tokenized sentence without stopwords to out_ns.csv
    writersRandom[1].writerow(negLineTokNs) # write to selected file without stopwords

    # randomly choose dataset for positive sentence (train, val, test)
    writersRandom = chooseDataSet() # returns tuple (writer, writerNs)
    writersRandom[0].writerow(posLineTok) # write to selected file with stopwords

    posLineTokNs = removeStopWords(posLineTok) # remove stopwords from sentence
    writerOutNs.writerow(posLineTokNs) # write tokenized sentence without stopwords to out_ns.csv
    writersRandom[1].writerow(posLineTokNs) # write to selected file without stopwords

    # count += 1
    # if count > limit:
    #   break

# close all files
neg.close()
pos.close()
# negSample.close()
# posSample.close()
sw.close()
out.close()
train.close()
val.close()
test.close()
outNoStop.close()
trainNoStop.close()
valNoStop.close()
testNoStop.close()