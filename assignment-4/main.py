import sys
import os
import numpy as np
import pickle
import gensim
from tensorflow.keras import regularizers
from tensorflow import one_hot
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence as s
from tensorflow.keras.layers import Dense, Embedding, Flatten, Dropout
from tensorflow.keras import Sequential

# command line argument: path to folder containing data splits with labels
path = sys.argv[1]
# path = "C:\\Users\\xsusa\\repos\\msci-nlp-w22\\assignment-4\\data\\"

# setting up all paths assuming cwd directory is ...\assignment-4 ----------------
data_path = os.getcwd() + "\\data\\" # path for output files

# load word2vec model from assignment 3
os.chdir("..")
w2v_path = os.getcwd() + "\\assignment-3\\data\\w2v.model"

# load and preprocess all necessary input documents --------------------------------
# open the split data sets with label files
train_ns_file = open(path + 'train_ns.csv', 'r')
val_ns_file = open(path + 'val_ns.csv', 'r')
test_ns_file = open(path + 'test_ns.csv', 'r')
label_train_file = open(path + 'train_labels.csv', 'r')
label_val_file = open(path + 'val_labels.csv', 'r')
label_test_file = open(path + 'test_labels.csv', 'r')

# gather all data into lists of tokenized strings
train = train_ns_file.read().splitlines()
val = val_ns_file.read().splitlines()
test = test_ns_file.read().splitlines()
train_labels = label_train_file.read().splitlines()
val_labels = label_val_file.read().splitlines()
test_labels = label_test_file.read().splitlines()

# one-hot encoding of labels to [0, 1] or [1, 0]
train_labels = list(map(int, train_labels))
train_labels = one_hot(train_labels, depth=2)
val_labels = list(map(int, val_labels))
val_labels = one_hot(val_labels, depth=2)
test_labels = list(map(int, test_labels))
test_labels = one_hot(test_labels, depth=2)

t = Tokenizer()
t.fit_on_texts(train) # update the internal vocabulary
# save tokenizer for use in inference 
with open(data_path + 'tokenizer.pkl', 'wb') as handle:
    pickle.dump(t, handle, protocol=pickle.HIGHEST_PROTOCOL)
vocab_size = len(t.word_index) + 1 # add 1 because vocab indexing starts at 1, not 0

# integer encode the documents
train_encoded = t.texts_to_sequences(train)
val_encoded = t.texts_to_sequences(val)
test_encoded = t.texts_to_sequences(test)

# gather lists of sentence lengths from training set and find the nth percentile
n_percentile = 85 # percent
train_sequence_lengths = [len(i) for i in train_encoded]
max_length = int(np.percentile(train_sequence_lengths, n_percentile))
print('Max sentence length: ', max_length)

# standardize the sequence lengths by padding or truncating
train_encoded = s.pad_sequences(train_encoded, maxlen=max_length, padding='post', truncating='post')
val_encoded = s.pad_sequences(val_encoded, maxlen=max_length, padding='post', truncating='post')
test_encoded = s.pad_sequences(test_encoded, maxlen=max_length, padding='post', truncating='post')

# Word2Vec(vocab=42123, vector_size=100, alpha=0.025)
w2v_model = gensim.models.Word2Vec.load(w2v_path)
# get keyed vectors (become immutable vectors)
word_vectors = w2v_model.wv
# dimension of each word vector (1x100)
emb_dimension = len(word_vectors.get_vector(0))

# create an embedding matrix to feed into the Embedding layer
embedding_matrix = np.zeros((vocab_size, emb_dimension))
for word, i in t.word_index.items(): # the dictionary key: values e.g. ('book', 1)
    try:
        embedding_vector = word_vectors.get_vector(word)
    except KeyError:
        # print("word ", word, " is not present in w2v model\n")
        continue
    embedding_matrix[i] = embedding_vector

# Embedding layer: first hidden layer ----------------------------
# trainable = false so that they remain fixed during training
embedding_layer = Embedding(
    input_dim=vocab_size,
    output_dim=emb_dimension,
    weights=[embedding_matrix],
    input_length=max_length, # padding method
    trainable=False,
)

# Build the models -----------------------------------------------
# testing parameters 
activations = ['relu', 'sigmoid', 'tanh'] # set of activation functions
num_hidden_nodes = 10 # number of nodes in hidden layer 
num_output_nodes = 2 # binary output of one hot vector
lambda_l2 = 0.01 # L2 regularization strength for matrices W and U
drop_out_rates = [0.5, 0.2, 0.05] # set input units to 0 with probability drop out rate

# create a dense layer U between output layer and hidden layer
u_dense_layer = Dense(
    units=num_output_nodes,
    activation="softmax",
    kernel_regularizer=regularizers.l2(l2=lambda_l2),
)

# create dict of dense layer Ws with each activation function
w_dense_layers = {}
for a in activations:
    w_dense_layers[a] = Dense(
        units=num_hidden_nodes,
        activation=a,
        kernel_regularizer=regularizers.l2(l2=lambda_l2),
    )

# build the feed forward nn model variations
models = {} # nested dictionary for activations and drop out rates
for a in activations:
    models[a] = {} 
    for d in drop_out_rates:
        models[a][d] = Sequential()
        models[a][d].add(embedding_layer)
        models[a][d].add(Flatten())
        models[a][d].add(w_dense_layers[a])
        models[a][d].add(Dropout(rate=d))
        models[a][d].add(u_dense_layer)
        models[a][d].compile(loss='binary_crossentropy', metrics=['accuracy'])

# all models have the same dimensions and number of parameters
print(models['relu'][0.5].summary())

# train the models ---------------------------------------------
print("\n\ntraining.....")

for a in activations:
    for d in drop_out_rates:
        print("\n\ntraining ffnn with activation func: ", a, "\tdrop out rate: ", str(d))
        print(models[a][d])
        models[a][d].fit(
            x=train_encoded,
            y=train_labels,
            batch_size=30,
            epochs=10,
            validation_data=(val_encoded, val_labels),
        )

# test the models  -----------------------------------------------
print("\n\ntesting.....\n\n")
best_models = {} # top model for every activation function
for a in activations:
    max_score = 0
    for d in drop_out_rates:
        print("\n\ntesting ffnn with activation func: ", a, "\tdrop out rate: ", str(d))
        print(models[a][d])
        score = models[a][d].evaluate(test_encoded, test_labels)
        print("activation func: ", a, "\tdrop out rate: ", str(d), "\t", models[a][d].metrics_names[1], str(score[1]*100))
        if score[1] > max_score:
            max_score = score[1]
            best_models[a] = models[a][d]
            print("best model - activation: ", a, "\tdrop out rate: ", str(d), "")

# save the best models
print("\n")
print(best_models) 
for m in best_models:
    print("\nfor activation ", m, " saving ", best_models[m])
    best_models[m].save(data_path + 'nn_' + m + '.model')
