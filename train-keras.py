#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
from tensorflow.keras import backend as K
from tensorflow.keras.applications import ResNet50

# Parameters
# ==================================================
checkpoint_directory = "resnet_nonstatic"
embedding = "word2vec"
embedding_dim = 300
model = "resnet"
filter_sizes = [3,4,5]
num_filters = 128
num_channels = 2

evaluate_every = 100
checkpoint_every = 100
num_checkpoints = 2

dropout_keep_prob = 0.5
l2_reg_lambda = 0.0
batch_size = 64
num_epochs = 200

def preprocess():
    # Data Preparation
    # ==================================================

    # Load data
    print("Loading data...")
    x_train, y_train = data_helpers.load_sst_binary('./data/sst-binary/stsa.binary.train')
    x_dev, y_dev = data_helpers.load_sst_binary('./data/sst-binary/stsa.binary.test')

    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_train])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x_train = np.array(list(vocab_processor.fit_transform(x_train)))
    x_dev = np.array(list(vocab_processor.fit_transform(x_dev)))

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, y_train, vocab_processor, x_dev, y_dev

def train(x_train, y_train, vocab_processor, x_dev, y_dev):
    # train using keras : resnet
    # ==================================================

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=True,
          log_device_placement=False)
        sess = tf.Session(config=session_conf)
        print('backend',K.backend())
        #K.set_session(sess) # connect Keras backend

        with sess.as_default():
            with tf.device('/gpu:0'):
                model = ResNet50(include_top=True, weights=None, classes=y_train.shape[1], input_shape=(x_train.shape[1], embedding_dim, 1))

            # Define Training procedure
            # keras implementation model compile
            optimizer = tf.keras.optimizers.Adam(0.001)

            model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            #model.summary()
            
            # global_step = tf.Variable(0, name="global_step", trainable=False)
            # optimizer = tf.train.AdamOptimizer(1e-3)
            # grads_and_vars = optimizer.compute_gradients(cnn.loss)
            # train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Output directory for models and summaries
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", checkpoint_directory))
            print("Writing to {}\n".format(out_dir))

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.all_variables())

            sequence_length = x_train.shape[1]

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            with tf.device('/cpu:0'), tf.name_scope("embedding"):
                W = tf.Variable(
                    tf.random_uniform([len(vocab_processor.vocabulary_), embedding_dim], -1.0, 1.0),
                    name="W",
                    trainable=True)

            # Initialize all variables
            sess.run(tf.initialize_all_variables())

            if embedding == "word2vec":
               # initial matrix with random uniform
                initW = np.random.uniform(-0.25,0.25,(len(vocab_processor.vocabulary_), embedding_dim))
                # load any vectors from the word2vec
                print("Embed word using {}\n".format(embedding))
                with open("./embedding/GoogleNews-vectors-negative300.bin", "rb") as f:
                    header = f.readline()
                    vocab_size, layer1_size = map(int, header.split())  # 3000000, 300
                    binary_len = np.dtype('float32').itemsize * layer1_size # 1200
                    # print(vocab_size, layer1_size)
                    for line in range(vocab_size):
                        # print(line)
                        word = []
                        while True:
                            ch = f.read(1).decode('latin-1')
                            if ch == ' ':
                                word = ''.join(word)
                                break
                            if ch != '\n':
                                word.append(ch)
                            else:
                                print('else: ', word, ch)
                        # print(word)
                        idx = vocab_processor.vocabulary_.get(word)
                        # print("value of idx is" + str(idx));
                        if idx != 0:
                            # print("came to if")
                            initW[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                        else:
                            # print("came to else");
                            f.read(binary_len)
                W.assign(initW)
                print("Ended")

            def train_step(x_batch, y_batch, step):
                """
                A single training step
                """
                x_batch = np.asarray(x_batch)
                x_batch = tf.expand_dims(tf.nn.embedding_lookup(W, x_batch), -1)
                y_batch = tf.argmax(np.array(y_batch), axis=1)
                y_batch = tf.reshape(y_batch, [-1,1])
                
                loss, accuracy = model.train_on_batch(x_batch, y_batch)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

            def dev_step(x_batch, y_batch, step):
                """
                Evaluates model on a dev set
                """
                x_batch = np.asarray(x_batch)
                x_batch = tf.expand_dims(tf.nn.embedding_lookup(W, x_batch), -1)
                y_batch = tf.argmax(np.array(y_batch), axis=1)
                y_batch = tf.reshape(y_batch, [-1,1])

                loss, accuracy = model.test_on_batch(x_batch, y_batch)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), batch_size, num_epochs)
            # Training loop. For each batch...
            current_step = 0
            #with tf.device('/gpu:0'):
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                current_step += 1
                train_step(x_batch, y_batch, current_step)
                #current_step = tf.train.global_step(sess, global_step)
                if current_step % evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, current_step)
                    print("")
                if current_step % checkpoint_every == 0:
                    #path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    path = os.path.join(checkpoint_prefix), current_step
                    model.save(path)
                    print("Saved model checkpoint to {}\n".format(path))

def main(argv=None):
    x_train, y_train, vocab_processor, x_dev, y_dev = preprocess()
    train(x_train, y_train, vocab_processor, x_dev, y_dev)

if __name__ == '__main__':
    tf.app.run()
