import sys
import os
import pickle
from tensorflow.keras.preprocessing import sequence as s
from tensorflow.keras import models

# command line arguments accepted:
# 1) path to .txt file with test sentences
# 2) type of classifier to use ("relu", "sigmoid", "tanh")
path_test = sys.argv[1]
clfName = sys.argv[2]

# setting up the file path for saved models assuming cwd directory is ...\assignment-4
tokenizer_path = os.getcwd() + "\\data\\tokenizer.pkl"
model_path = os.getcwd() + "\\data\\" + "nn_" + clfName + ".model"

# open the test file with raw sentences
test_file = open(path_test, 'r')
# read the test file lines into a list
test = test_file.read().splitlines()

# load the tokenizer fitted on the training set
with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

# integer encode the test sentences
test_encoded = tokenizer.texts_to_sequences(test)

# standardize test sentence lengths
max_length = 11 # maximum sentence length defined during training
test_encoded = s.pad_sequences(test_encoded, maxlen=max_length, padding='post', truncating='post')

# load the ffnn model
model = models.load_model(model_path)

# predict sentiment of test sentences (pos or neg)
prediction = model.predict(test_encoded)
prediction_round = prediction.round()

# one hot vector mapping: 0 -> [1 0], 1 -> [0 1]
for i in range(len(test)):
    print("sentence:\t", test[i])
    if prediction_round[i][0] == 0:
        print("prediction: ", prediction[i], "\tsentiment: positive\n")
    else:
        print("prediction: ", prediction[i], "\tsentiment: negative\n")