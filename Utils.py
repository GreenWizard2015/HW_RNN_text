import tensorflow as tf
import numpy as np
from _collections import defaultdict
import matplotlib.pyplot as plt
import itertools

def _setupRandomSeed():
  SEED = [None]
  def setSeed(seed=None):
    seed = SEED[0] if seed is None else seed
    SEED[0] = seed
    
    np.random.seed(seed)
    return
  
  return setSeed

setupRandomSeed = _setupRandomSeed()

def setup(MAX_GPU_MEMORY, RANDOM_SEED):
  gpus = tf.config.experimental.list_physical_devices('GPU')
  tf.config.experimental.set_virtual_device_configuration(
    gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=MAX_GPU_MEMORY)]
  )
  setupRandomSeed(RANDOM_SEED)

def saveMetrics(metrics, filepath, startEpoch=0):
  collectedData = defaultdict(dict)
  for dataName, values in metrics.items():
    name = dataName.replace('val_', '')
    metricKind = 'test' if dataName.startswith('val_') else 'train'
    collectedData[name][metricKind] = list(values)
    
  for name, data in collectedData.items():
    plt.clf()
    fig = plt.figure()
    axe = fig.subplots(ncols=1, nrows=1)
    for nm, values in data.items():
      axe.plot(values[startEpoch:], label=nm)
      
    axe.title.set_text(name)
    axe.set_ylabel(name)
    axe.set_xlabel('epoch')
    axe.legend(loc='upper left')
    fig.savefig(filepath('%s.png' % name))
    plt.close(fig)
    
  return

def plot_confusion_matrix(
  cm,
  target_names,
  saveTo,
  title='Confusion matrix',
  onlyErrors=False
):
  plt.clf()
  accuracy = np.trace(cm) / float(np.sum(cm))
  misclass = 1 - accuracy
  # mask out diagonal
  if onlyErrors:
    for i in range(cm.shape[0]):
      cm[i, i] = 0

  plt.figure(figsize=(8, 6))
  plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
  plt.title('%s (accuracy=%0.4f; misclass=%0.4f)' % (title, accuracy, misclass))
  plt.colorbar()

  if target_names is not None:
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)

  thresh = cm.max() / 2.0
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    text = str(cm[i, j])
    if onlyErrors:
      if (cm[i, j] <= 0.0) or (i == j):
        text = ''
    
    color = "white" if cm[i, j] > thresh else "black"
    plt.text(j, i, text, horizontalalignment="center", color=color)
    continue
  
  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.savefig(saveTo)
  plt.close()
  return
