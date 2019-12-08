#-*- coding: utf-8 -*- 
import tensorflow as tf
from tensorflow.python.keras.layers import *
from tensorflow.python.keras import Model, Input
from tqdm import tqdm
import json


VOCAB_SIZE = 13832

# word --> vector dictionary
def set_dict():
   '''
   mapper = {}
   with open("./data/wiki-news-300d-1M-subword.vec", 'r', encoding='utf-8') as f: 
      f.readline()
      
      for line in tqdm(f.readlines()):
         splited = line.split()
         mapper[splited[0]] = splited[1:]


   import json
   json = json.dumps(mapper)
   with open("./data/mapper.json","w") as f:
      f.write(json)
      f.close()

   raise()
   '''
   t = json_load("./data/mapper.json")
   print(f"************{type(t)}**********")
   raise()
   return t


# convert string input to vector
def input_to_vector(input_string, mapper):
   vector = np.zeros((VOCAB_SIZE, 300))
   splited = input_string.split()
   label, data = int(splited[0]), splited[1:]
   for idx, word in enumerate(data):
      vector[idx] = np.array(mapper[word])

   return label, vector

# return the datasets train, test, dev
def preprocess(batch_size):
   mapper = set_dict()
   phases = ["train", "test", "dev"]
   prefix = "./data/stsa.binary."
   datasets = []

   max_ = 0

   for phase in phases:
      fname = prefix + phase
      with open(fname, 'r', 'utf-8') as f:
         labels = []
         vectors = []
         for line in f.readlines()[:128]:
            label, vector = input_to_vector(line, mapper)
            labels.append(label)
            vectors.append(vector)
            if max_ < len(vectors):
               max_ = len(vectors)

      vectors = np.array(vectors)
      labels = np.array(labels)

      dataset = tf.data.Dataset.from_tensor_slices((vectors, labels)).shuffle(BATCH_SIZE).batch(BATCH_SIZE)
      datasets.append(dataset)

   global VOCAB_SIZE
   VOCAB_SIZE = max_

   return datasets



class TextCNN(Model):
   """Fusion class."""

   def __init__(self, **kwargs):
      super(TextCNN, self).__init__(**kwargs)

      self.conv1 = Conv(1, (3, 300))
      self.maxPooing1 = MaxPoolin1D(strides=2)
      self.flatten1 = flatten()
      self.conv2 = Conv(1, (4, 300))
      self.maxPooing2 = MaxPoolin1D(strides=2)
      self.flatten2 = flatten()
      self.conv3 = Conv(1, (5, 300))
      self.maxPooing3 = MaxPoolin1D(strides=2)
      self.flatten3 = flatten()

      self.fc = dense(2, use_bias=False, activation='softmax')


   def call(self, inputs):

      x1 = self.conv1(inputs)
      x1 = self.maxPooing1(x1)
      x1 = self.flatten1(x1)

      x2 = self.conv2(inputs)
      x2 = self.maxPooing2(x2)
      x2 = self.flatten2(x2)

      x3 = self.conv3(inputs)
      x3 = self.maxPooing3(x3)
      x3 = self.flatten3(x3)

      x = tf.concat([x1, x2, x3], axis=-1)
      x = self.fc(x)

      return x






# dataset configure
train_dataset, test_dataset, valid_dataset = preprocess(64)


# model configure
model = TextCNN()


# train parameters
optimizer = tf.keras.optimizers.Adamax(lr=start_lr)
loss = 'categorical_crossentropy'
model.compile(optimizer=optimizer, loss=loss, metrics=tf.keras.metrics.CategoricalAccuracy())

model.build(input_shape=(None, VOCAB_SIZE, 300))
model.summary()


# train model
history = model.fit(train_dataset, \
               epochs=200, \
               validation_data=valid_dataset, \
               verbose=1)


# test model
result = model.evaluate(test_dataset, verbose=1)