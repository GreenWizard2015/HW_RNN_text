#!/usr/bin/env python
# -*- coding: utf-8 -*-
import Utils
Utils.setup(MAX_GPU_MEMORY=4 * 1024, RANDOM_SEED=1234)

import os
import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np
from CTextPreprocessor import CTextPreprocessor
from CTextGenerator import CTextGenerator

BATCH_SIZE = 128
MAX_EPOCHS = 1000
PREDICTION_LEN = 300
INPUT_FILE = "dataset.txt"

TEST_PHRASES = [
  'оскорбительно притворно-сладкое ',
  'Анна Павловна ',
  'Но ведь ',
  'надо прервать все эти ',
  'послала за сыном и ',
]
################################

def fromFile(filename, encoding='utf8'):
  with open(filename, 'r', encoding=encoding) as f:
    return f.read()
  return

dataset = CTextPreprocessor(fromFile(INPUT_FILE), r'[^a-zа-я0-9\s\.,\!\(\)\-\ё]')

splitInd = int(len(dataset.text) * 0.8)
trainText = dataset.text[:splitInd]
testText = dataset.text[splitInd:]

def argmaxSampling(text, probes):
  return np.argmax(probes)

def generateText(model, seed, predictionLen, sampling=argmaxSampling, maxSeedLen=None):
  maxSeedLen = predictionLen if maxSeedLen is None else maxSeedLen
  while len(seed) < predictionLen:
    pred = model([
      np.array([ dataset.encode(seed[-maxSeedLen:]) ])
    ]).numpy()[0]
    
    if len(pred.shape) == 2:
      pred = pred[-1]
      
    seed += dataset.decode(sampling(seed, pred))
    continue
  return seed

def createSequentialRNNModel(
  NChars, return_sequences,
  RNNLayer=layers.LSTM, EmbeddingSize=2, HIDDEN_SIZE=128
):
  L = tf.keras.layers
  res = characters = L.Input(shape=(None,))
  
  res = L.Embedding(NChars, EmbeddingSize)(res)
  res = L.Activation('tanh')(res)

  res = RNNLayer(HIDDEN_SIZE, return_sequences=True)(res)
  res = RNNLayer(HIDDEN_SIZE, return_sequences=True)(res)
  res = RNNLayer(HIDDEN_SIZE, return_sequences=return_sequences)(res)
  
  res = L.Dense(NChars, activation='softmax')(res)
  
  return tf.keras.Model(inputs=[characters], outputs=[res])

##########################

def sampleTopK(K, onlyFirstCharacter=False):
  def f(text, x):
    if onlyFirstCharacter and not (' ' == text[-1]):
      return np.argmax(x)
    
    topK = np.argsort(x)[-K:]
    v = x[topK] + 0.01
    return np.random.choice(topK, p=v / v.sum())
  return f

class COnEpochEndCallback(tf.keras.callbacks.Callback):
  def __init__(self, callback):
    super().__init__()
    self._callback = callback
    return

  def on_epoch_end(self, epoch, logs=None):
    self._callback()
    return
  
def trainAndTest(
  return_sequences,
  SEQUENCE_LEN, INFER_SEQUENCE_LEN=None,
  tag=None,
  RNNLayer=layers.LSTM
):
  INFER_SEQUENCE_LEN = SEQUENCE_LEN if INFER_SEQUENCE_LEN is None else INFER_SEQUENCE_LEN
  
  Utils.setupRandomSeed()
  modelName = '%s%s-%d' % (
    '' if tag is None else tag + '-',
    'seq' if return_sequences else 'single',
    SEQUENCE_LEN
  )
  print('Start training %s' % modelName)
  FOLDER = os.path.join(os.path.dirname(__file__), 'output')
  filepath = lambda *x: os.path.join(FOLDER, *x)
  os.makedirs(FOLDER, exist_ok=True)
  
  model = createSequentialRNNModel(dataset.N_chars, return_sequences=return_sequences, RNNLayer=RNNLayer)
  model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=tf.optimizers.Adam(learning_rate=1e-3, clipnorm=1.)
  )
  model.summary()

  def debugModel():
    for testSeed in TEST_PHRASES:
      text = generateText(
        model, testSeed,
        predictionLen=PREDICTION_LEN, maxSeedLen=SEQUENCE_LEN
      )
      print(text)
    return
  
  history = model.fit(
    CTextGenerator(
      trainText,
      batch_size=BATCH_SIZE,
      seqLen=SEQUENCE_LEN,
      encoder=dataset.encode,
      return_sequences=return_sequences
    ),
    validation_data=CTextGenerator(
      testText,
      batch_size=BATCH_SIZE,
      seqLen=SEQUENCE_LEN,
      encoder=dataset.encode,
      return_sequences=return_sequences
    ),
    epochs=MAX_EPOCHS,
    verbose=2,
    callbacks=[
      tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', mode='min', patience=5,
        restore_best_weights=True
      ),
      COnEpochEndCallback(debugModel)
    ]
  ).history
  Utils.saveMetrics(history, filepath=lambda x: filepath('%s-%s' % (modelName, x)))
  
  # testing model
  with open(filepath('%s.txt' % modelName), 'w') as f:
    def testSampling(title, maxSeedLen, sampling):
      f.write('%s\n' % title)
      f.write(('=' * 50) + '\n')
      for testSeed in TEST_PHRASES:
        f.write("Seed: %s\n" % testSeed)
        text = generateText(
          model, testSeed,
          predictionLen=PREDICTION_LEN, maxSeedLen=maxSeedLen,
          sampling=sampling
        )
        f.write(text + '\n\n')
      return
    
    seqLen = [SEQUENCE_LEN, INFER_SEQUENCE_LEN]
    if return_sequences:
      seqLen.append(None)
    
    for maxSeedLen in set(seqLen):
      testSampling('Sampling by argmax (Length = %s)' % str(maxSeedLen), maxSeedLen, argmaxSampling)
      testSampling(
        'Sampling by random top-5 (Length = %s)' % str(maxSeedLen),
        maxSeedLen=maxSeedLen,
        sampling=sampleTopK(K=5)
      )
      testSampling(
        'Sampling by random top-5 only first character of word (Length = %s)' % str(maxSeedLen),
        maxSeedLen=maxSeedLen,
        sampling=sampleTopK(K=5, onlyFirstCharacter=True)
      )
      continue

  return

#############
trainAndTest(return_sequences=False, SEQUENCE_LEN=10)
trainAndTest(return_sequences=False, SEQUENCE_LEN=20)
 
trainAndTest(return_sequences=True, SEQUENCE_LEN=10, INFER_SEQUENCE_LEN=10)
trainAndTest(return_sequences=True, SEQUENCE_LEN=20, INFER_SEQUENCE_LEN=10)
trainAndTest(return_sequences=True, SEQUENCE_LEN=40, INFER_SEQUENCE_LEN=10)
 
trainAndTest(return_sequences=True, SEQUENCE_LEN=10, INFER_SEQUENCE_LEN=10, RNNLayer=layers.GRU, tag='gru') 
trainAndTest(return_sequences=True, SEQUENCE_LEN=40, INFER_SEQUENCE_LEN=10, RNNLayer=layers.GRU, tag='gru')
