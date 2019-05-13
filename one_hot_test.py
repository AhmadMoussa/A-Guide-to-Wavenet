from __future__ import print_function

import librosa
import numpy as np
import tensorflow as tf
from sys import stdin,stdout,stderr,maxsize


#np.set_printoptions(threshold = maxsize)

batch_size = 1
quantization_channels = 256

# returns a tensorflow one_hot encoding
def _one_hot(input_batch):
    encoded = tf.one_hot(input_batch, depth = quantization_channels, dtype = tf.float32)
    shape = [batch_size, -1, quantization_channels]
    encoded = tf.reshape(encoded, shape)
    return encoded

# load an audio from librosa
def load_audio():
    filename = librosa.util.example_audio_file()
    waveform = librosa.load(filename)
    waveform = waveform[0].reshape(-1,1)
    return waveform

print(load_audio())

'''
one_hot = _one_hot(load_audio())
sess = tf.InteractiveSession()

one_hot_prnt = tf.Print(one_hot, [one_hot], "One hot encoding:", summarize = 25700)
evaluator = tf.add(one_hot_prnt, one_hot_prnt)

evaluator.eval()
'''
