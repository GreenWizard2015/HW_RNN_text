import tensorflow as tf
import numpy as np
import math

class CTextGenerator(tf.keras.utils.Sequence):
  def __init__(self, text, batch_size, seqLen, encoder, return_sequences=False):
    super().__init__()
    self._text = text
    self._encoder = encoder
    self._batchSize = batch_size
    self._seqLen = seqLen
    self._returnSequences = return_sequences
    
    maxIndex = len(text) - seqLen
    self._indices = np.arange(math.ceil(maxIndex / batch_size) * batch_size) % maxIndex
    return
  
  def on_epoch_end(self):
    np.random.shuffle(self._indices)
    return
  
  def __getitem__(self, batchIndex):
    current = batchIndex * self._batchSize
    indices = self._indices[current:current+self._batchSize]
    
    X = []
    Y = []
    for i in indices:
      X.append(self._encoder(self._text[i:i+self._seqLen]))
      
      if self._returnSequences:
        target = self._text[i+1:i+self._seqLen+1]
      else:
        target = self._text[i+self._seqLen]
        
      Y.append(self._encoder(target))
      continue
    return(np.array(X), np.array(Y).reshape(len(Y), -1, 1))

  def __len__(self):
    return len(self._indices) // self._batchSize