import re

class CTextPreprocessor:
  def __init__(self, text, validCharactersRE):
    self._text = re.sub(
      r'\s+',
      ' ',
      re.sub(validCharactersRE, ' ', text.lower())
    )
    
    self._chars = list(sorted(set(self._text)))
    self._char2index = dict((c, i) for i, c in enumerate(self._chars))
    self._index2char = dict((i, c) for i, c in enumerate(self._chars))
    return
  
  def encode(self, text):
    return [self._char2index[c] for c in text.lower()]
  
  def decode(self, ind):
    return self._index2char[ind]
  
  @property
  def N_chars(self):
    return len(self._chars)
  
  @property
  def text(self):
    return self._text