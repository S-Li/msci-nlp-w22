import sys
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# process command line argument for path to split data sets
path = sys.argv[1] # skip first argument which is script name
# path = 'C:\\Users\\xsusa\\repos\\msci-nlp-w22\\assignment-2\\data\\'

# create output files for fitted vectorizer objects
vUniOutFile = open(path + 'v_uni.pkl', 'wb')
vBiOutFile = open(path + 'v_bi.pkl', 'wb')
vUniBiOutFile = open(path + 'v_uni_bi.pkl', 'wb')
vUniNsOutFile = open(path + 'v_uni_ns.pkl', 'wb')
vBiNsOutFile = open(path + 'v_bi_ns.pkl', 'wb')
vUniBiNsOutFile = open(path + 'v_uni_bi_ns.pkl', 'wb')

# create output files for trained classifier objects
mnbUniOutFile = open(path + 'mnb_uni.pkl', 'wb')
mnbBiOutFile = open(path + 'mnb_bi.pkl', 'wb')
mnbUniBiOutFile = open(path + 'mnb_uni_bi.pkl', 'wb')
mnbUniNsOutFile = open(path + 'mnb_uni_ns.pkl', 'wb')
mnbBiNsOutFile = open(path + 'mnb_bi_ns.pkl', 'wb')
mnbUniBiNsOutFile = open(path + 'mnb_uni_bi_ns.pkl', 'wb')

# open the split data sets with label files
train = open(path + 'train.csv', 'r')
val = open(path + 'val.csv', 'r')
test = open(path + 'test.csv', 'r')
trainNoStop = open(path + 'train_ns.csv', 'r')
valNoStop = open(path + 'val_ns.csv', 'r')
testNoStop = open(path + 'test_ns.csv', 'r')
labelTrain = open(path + 'train_labels.csv', 'r')
labelVal = open(path + 'val_labels.csv', 'r')
labelTest = open(path + 'test_labels.csv', 'r')

# gather all data into list objects
trainLines = train.read().splitlines()
valLines = val.read().splitlines()
testLines = test.read().splitlines()
trainNoStopLines = trainNoStop.read().splitlines()
valNoStopLines = valNoStop.read().splitlines()
testNoStopLines = testNoStop.read().splitlines()
labelTrainLines = labelTrain.read().splitlines()
labelValLines = labelVal.read().splitlines()
labelTestLines = labelTest.read().splitlines()
# trainLength = len(train)
# train = train[:(trainLength)]
# labelTrain = labelTrain[:(trainLength)]

# create 6 vectorizers to generate frequency matrices
# of unique features based on unigrams, bigrams, and both
uniVectorizer = CountVectorizer(ngram_range=(1,1)) # only unigrams
biVectorizer = CountVectorizer(ngram_range=(2,2)) # only bigrams
uniBiVectorizer = CountVectorizer(ngram_range=(1,2)) # unigrams and bigrams
uniVectorizerNs = CountVectorizer(ngram_range=(1,1)) # only unigrams, no sw
biVectorizerNs = CountVectorizer(ngram_range=(2,2)) # only bigrams, no sw
uniBiVectorizerNs = CountVectorizer(ngram_range=(1,2)) # unigrams and bigrams, no ns

# sort training data into frequency matrices
uniTrainCount = uniVectorizer.fit_transform(trainLines) 
biTrainCount = biVectorizer.fit_transform(trainLines)
uniBiTrainCount = uniBiVectorizer.fit_transform(trainLines)
uniTrainCountNs = uniVectorizerNs.fit_transform(trainNoStopLines) 
biTrainCountNs = biVectorizerNs.fit_transform(trainNoStopLines)
uniBiTrainCountNs = uniBiVectorizerNs.fit_transform(trainNoStopLines)

# sort validation data into frequency matrices
uniValCount = uniVectorizer.transform(valLines) 
biValCount = biVectorizer.transform(valLines)
uniBiValCount = uniBiVectorizer.transform(valLines)
uniValCountNs = uniVectorizerNs.transform(valNoStopLines) 
biValCountNs = biVectorizerNs.transform(valNoStopLines)
uniBiValCountNs = uniBiVectorizerNs.transform(valNoStopLines)

# sort test data into frequency matrices
uniTestCount = uniVectorizer.transform(testLines) 
biTestCount = biVectorizer.transform(testLines)
uniBiTestCount = uniBiVectorizer.transform(testLines)
uniTestCountNs = uniVectorizerNs.transform(testNoStopLines) 
biTestCountNs = biVectorizerNs.transform(testNoStopLines)
uniBiTestCountNs = uniBiVectorizerNs.transform(testNoStopLines)

# measure accuracy of classifier 
# input: classifier, val or test frequency matrices, labels 
# output: accuracy value
def getClfAccuracy(clf, freqMatrix, labels):
    accuracy = clf.score(freqMatrix, labels)
    # print(accuracy)
    return accuracy

# tuning alpha hyperparameter for best prediction accuracy
# default parameters: {'alpha': 1.0, 'class_prior': None, 'fit_prior': True}
# input: training set and validation set frequency matrices
# output: MNB classifier with highest-scoring alpha value
def getTunedClf(trainCount, valCount):
    paramGrid = np.arange(0.1, 5.2, 0.5).tolist() # possible alpha values
    maxScore = 0
    score = 0
    bestAlpha = 0
    for p in paramGrid:
        # fit model to the training set at a specific alpha value
        model = MultinomialNB(alpha=p)
        model.fit(trainCount, labelTrainLines)
        # test accuracy using the specific validation set
        score = getClfAccuracy(model, valCount, labelValLines)
        if score > maxScore:
            maxScore = score
            bestAlpha = p
    model = MultinomialNB(alpha=bestAlpha)
    model.fit(trainCount, labelTrainLines)
    return (model)


# train the MNB classifiers using the training set and tune
# using the corresponding validation sets

# classifier for unigrams with stopwords
mnbUni = getTunedClf(uniTrainCount, uniValCount)
# classifier for bigrams with stopwords
mnbBi = getTunedClf(biTrainCount, biValCount)
# classifier for unigrams + bigrams with stopwords
mnbUniBi = getTunedClf(uniBiTrainCount, uniBiValCount)
# classifier for unigrams without stopwords
mnbUniNs = getTunedClf(uniTrainCountNs, uniValCountNs)
# classifier for bigrams without stopwords
mnbBiNs = getTunedClf(biTrainCountNs, biValCountNs)
# classifier for unigrams + bigrams without stopwords
mnbUniBiNs = getTunedClf(uniBiTrainCountNs, uniBiValCountNs)

# evaluate the classifiers against test data
accuracyUni = getClfAccuracy(mnbUni, uniTestCount, labelTestLines)
accuracyBi = getClfAccuracy(mnbBi, biTestCount, labelTestLines)
accuracyUniBi = getClfAccuracy(mnbUniBi, uniBiTestCount, labelTestLines)
accuracyUniNs = getClfAccuracy(mnbUniNs, uniTestCountNs, labelTestLines)
accuracyBiNs = getClfAccuracy(mnbBiNs, biTestCountNs, labelTestLines)
accuracyUniBiNs = getClfAccuracy(mnbUniBiNs, uniBiTestCountNs, labelTestLines)

# Fill in  results table
results = [
    {
        "stopwords removed" : "yes",
        "text features" : "unigrams",
        "accuracy" : accuracyUniNs,
    },
    {
        "stopwords removed" : "yes",
        "text features" : "bigrams",
        "accuracy" : accuracyBiNs,
    },
    {
        "stopwords removed" : "yes",
        "text features" : "unigrams + bigrams",
        "accuracy" : accuracyUniBiNs,
    },
    {
        "stopwords removed" : "no",
        "text features" : "unigrams",
        "accuracy" : accuracyUni,
    },
    {
        "stopwords removed" : "no",
        "text features" : "bigrams",
        "accuracy" : accuracyBi,
    },
    {
        "stopwords removed" : "no",
        "text features" : "unigrams + bigrams",
        "accuracy" : accuracyUniBi,
    },
]

for r in results:
    print(r)

# dump the fitted vectorizers into pkl files
pickle.dump(uniVectorizer, vUniOutFile)
pickle.dump(biVectorizer, vBiOutFile)
pickle.dump(uniBiVectorizer, vUniBiOutFile)
pickle.dump(uniVectorizerNs, vUniNsOutFile)
pickle.dump(biVectorizerNs, vBiNsOutFile)
pickle.dump(uniBiVectorizerNs, vUniBiNsOutFile)

# dump the trained classifiers into pkl files
pickle.dump(mnbUni, mnbUniOutFile)
pickle.dump(mnbBi, mnbBiOutFile)
pickle.dump(mnbUniBi, mnbUniBiOutFile)
pickle.dump(mnbUniNs, mnbUniNsOutFile)
pickle.dump(mnbBiNs, mnbBiNsOutFile)
pickle.dump(mnbUniBiNs, mnbUniBiNsOutFile)


