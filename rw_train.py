# dependencies
import os
import time 
import tqdm
import io

import numpy as np, pandas as pd
import pickle
import random

import tensorflow as tf
from tensorflow.keras import layers

from PixelCorpora import PixelCorpus, PixelCorpusRW

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-train", help="Use text data from file TRAIN to train the model", default=None) # changed to not required
parser.add_argument("-ds_ids", help="Metaspace dataset IDs for training", nargs = "+",  default=None)# added this one
parser.add_argument("-ind_name", help="Calculate model for ion or formula", default=None)
parser.add_argument("-fdr", help="FDR threshold", type=float, default= 0.1)
parser.add_argument("-pix_per", help="Pixel percentage considered in every dataset", type=float, default=0.5)
parser.add_argument("-int_per", help="Intensity percentage threshold in every dataset", type=float, default=0.5)
parser.add_argument("-output", help="Use file OUTPUT to save the resulting word vectors",type=str, default = 'test_output')
parser.add_argument("-window", help="Window size for pixel window (2D); int w will result in (2w+1)x(2w+1) window. default is 1 (3x3) window.", type=int, default=1)
parser.add_argument("-size", help="Set size of word vectors; default is 100", type=int, default=20)
parser.add_argument("-sample", help="Set threshold for occurrence of words. Those that appear with higher frequency in the training data will be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)", type=float, default=1e-3)
parser.add_argument("-hs", help="Use Hierarchical Softmax; default is 0 (not used)", type=int, default=0, choices=[0, 1])
parser.add_argument("-negative", help="Number of negative examples; default is 5, common values are 3 - 10 (0 = not used)", type=int, default=10)
parser.add_argument("-iter", help="Run more training iterations (default 5)", type=int, default=5)

parser.add_argument("-quan", help="Quantile parameter", type = float, default = 0)

parser.add_argument("-stride", help="Stride length of window, default is 1 (no stride)", type = int, default = 1)
parser.add_argument("-rw", help="Random walk version, default is 1 (on), else vanilla ion2vec", type = int, default=1)
parser.add_argument("-no_samples", help="# of sampled random walks pers window", type = int, default = 5)
parser.add_argument("-sentence_length", help ="Length of random walk", type = int, default = None)
parser.add_argument("-word_window", help="Window size in skip gram (1D).", type = int, default = 5)
    
args = parser.parse_args()

def extract_ind_name():
    if args.ind_name:
        with open(args.ind_name, 'rb') as remaining_ions:
            ind_name = pickle.load(remaining_ions)
        print('Creating vocab for train ions: ', ind_name)
    else: ind_name = None  
    return ind_name


ind_name = extract_ind_name()
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


if args.rw == 1:
    corpus = PixelCorpusRW(ds_ids = args.ds_ids, ds_dir = args.train, ind_name = ind_name, fdr_thresh = args.fdr,
                           pix_per = args.pix_per, int_per = args.int_per, window = args.window, quan = args.quan,
                           stride = args.stride, no_samples=args.no_samples, walk_length = args.sentence_length)
else:
    corpus = PixelCorpus(ds_ids = args.ds_ids, ds_dir = args.train, ind_name = ind_name, fdr_thresh = args.fdr,
                         pix_per = args.pix_per, int_per = args.int_per, window = args.window, quan = args.quan,
                         stride = args.stride)

ions2idx = corpus.get_ions2ids() # vocabulary
vocab_size = len(ions2idx) + 2

num_ns = args.negative 
SEED = 42 #random seed
window_size = args.word_window # text window
AUTOTUNE = tf.data.AUTOTUNE
embedding_dim = args.size


def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
    # Elements of each training example are appended to these lists.
    targets, contexts, labels = [], [], []

    # Build the sampling table for `vocab_size` tokens.
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

    # Iterate over all sequences (sentences) in the dataset.
    for sequence in tqdm.tqdm(sequences):

    # Generate positive skip-gram pairs for a sequence (sentence).
        positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
          sequence,
          vocabulary_size=vocab_size,
          sampling_table=sampling_table,
          window_size=window_size,
          negative_samples=0)

    # Iterate over each positive skip-gram pair to produce training examples
    # with a positive context word and negative samples.
        for target_word, context_word in positive_skip_grams:
            context_class = tf.expand_dims(
                tf.constant([context_word], dtype="int64"), 1)
            negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                true_classes=context_class,
                num_true=1,
                num_sampled=num_ns,
                unique=True,
                range_max=vocab_size,
                seed=seed,
                name="negative_sampling")

            # Build context and label vectors (for one target word)
            negative_sampling_candidates = tf.expand_dims(
            negative_sampling_candidates, 1)

            context = tf.concat([context_class, negative_sampling_candidates], 0)
            label = tf.constant([1] + [0]*num_ns, dtype="int64")

            # Append each element from the training example to global lists.
            targets.append(target_word)
            contexts.append(context)
            labels.append(label)

    return targets, contexts, labels


targets, contexts, labels = generate_training_data(corpus,
                            window_size= window_size,
                            num_ns=num_ns,
                            vocab_size=vocab_size,
                            seed = SEED)
 
targets = np.array(targets)
contexts = np.array(contexts)[:,:,0]
labels = np.array(labels)

print('\n')
print(f"targets.shape: {targets.shape}")
print(f"contexts.shape: {contexts.shape}")
print(f"labels.shape: {labels.shape}")

BATCH_SIZE = 1024
BUFFER_SIZE = 10000
dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)

class Word2Vec(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.target_embedding = layers.Embedding(vocab_size,
                                          embedding_dim,
                                          input_length=1,
                                          name="w2v_embedding")
        self.context_embedding = layers.Embedding(vocab_size,
                                           embedding_dim,
                                           input_length=num_ns+1)

    def call(self, pair):
        target, context = pair
        # target: (batch, dummy?)  # The dummy axis doesn't exist in TF2.7+
        # context: (batch, context)
        if len(target.shape) == 2:
            target = tf.squeeze(target, axis=1)
        # target: (batch,)
        word_emb = self.target_embedding(target)
        # word_emb: (batch, embed)
        context_emb = self.context_embedding(context)
        # context_emb: (batch, context, embed)
        dots = tf.einsum('be,bce->bc', word_emb, context_emb)
        # dots: (batch, context)
        return dots

#for later
def custom_loss(x_logit, y_true):
      return tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=y_true)


t0 = time.time()
with strategy.scope():
    model = Word2Vec(vocab_size, embedding_dim)
    model.compile(optimizer='adam',
                     loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                     metrics=['accuracy'],
                    run_eagerly=True)

    model.fit(dataset, epochs=20) # adapt epochs, add callbacks
t1 = time.time()
total = t1 - t0
print("Training time: ", total)

weights = model.get_layer('w2v_embedding').get_weights()[0]
vocab = ions2idx

out_v = io.open(f'vectors_{args.output}_rw.tsv', 'w', encoding='utf-8')
out_m = io.open(f'metadata_{args.output}_rw.tsv', 'w', encoding='utf-8')

for index, word in enumerate(vocab):
    #if index == 0:
    #    continue  # skip 0, 
    vec = weights[index]
    out_v.write('\t'.join([str(x) for x in vec]) + "\n")
    out_m.write(word + "\n")
out_v.close()
out_m.close()


