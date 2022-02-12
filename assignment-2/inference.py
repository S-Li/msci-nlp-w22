# inference.py classifies a given sentence as either positive or negative
# this file assumes the current working directory is the assignment folder
import sys
import os
import pickle

# collect command line arguments
testFilePath = sys.argv[1] # path to .txt file for evaluation
clfName = sys.argv[2] # type of classifier to use

# open the test file with raw sentences
testFile = open(testFilePath, 'r')
# read the test file lines into a list
testFileLines = testFile.read().splitlines()

# get path to assignment data folder
cwd = os.getcwd() # assignment folder
pathToClfs = cwd + "\\data\\"

# unload the fitted vectorizer and trained 
# classifier objects from .pkl files
if clfName == 'mnb_uni':
    vFile = open(pathToClfs + 'v_uni.pkl', 'rb')
    clfFile = open(pathToClfs + 'mnb_uni.pkl', 'rb')
elif clfName == 'mnb_bi':
    vFile = open(pathToClfs + 'v_bi.pkl', 'rb')
    clfFile = open(pathToClfs + 'mnb_bi.pkl', 'rb')
elif clfName == 'mnb_uni_bi':
    vFile = open(pathToClfs + 'v_uni_bi.pkl', 'rb')
    clfFile = open(pathToClfs + 'mnb_uni_bi.pkl', 'rb')
elif clfName == 'mnb_uni_ns':
    vFile = open(pathToClfs + 'v_uni_ns.pkl', 'rb')
    clfFile = open(pathToClfs + 'mnb_uni_ns.pkl', 'rb')
elif clfName == 'mnb_bi_ns':
    vFile = open(pathToClfs + 'v_bi_ns.pkl', 'rb')
    clfFile = open(pathToClfs + 'mnb_bi_ns.pkl', 'rb')
elif clfName == 'mnb_uni_bi_ns':
    vFile = open(pathToClfs + 'v_uni_bi_ns.pkl', 'rb')
    clfFile = open(pathToClfs + 'mnb_uni_bi_ns.pkl', 'rb')

v = pickle.load(vFile) # set the vectorizer
clf = pickle.load(clfFile) # set the classifier
vFile.close()
clfFile.close()

# sort test data into frequency matrix
testCount = v.transform(testFileLines)

# get prediction array from test matrix
labelPredictions = clf.predict(testCount)

# print out each line of text with its predicted label
for i in range(len(testFileLines)):
    print(testFileLines[i], labelPredictions[i])