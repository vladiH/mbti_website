import os, sys
import numpy as np
if '__file__' in globals():
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class MBTI:
    def __init__(self, name='Myers Briggs test'):
        self._name = name
        self._colors = ["#8805A8", "#D9005B", "#84E900", "#E2FA00"]
    
    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, mbti_label):
        self._labels = list(mbti_label)
    
    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, mbti_values, thresh=0.5):
        value = mbti_values.numpy()[0]
        mask = np.less(value,thresh)
        value = np.where(mask, 1.- value, value)
        self._values = list(np.round(value*100, decimals=2))

    @property
    def colores(self):
        return self._colors
    
    @property
    def name(self):
        return self._name