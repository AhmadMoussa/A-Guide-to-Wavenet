import tensorflow as tf
from audio_loader_and_encoder import lqe
from create_variable import create_variable
from causal_convolution_layer import _create_causal_layer

sess = tf.InteractiveSession()
loader = lqe()

path = "D:\\Users\\amm65\\Desktop\\Audio Sample Dataset\\Shroom_Drums from Another Planet\\Shroom_Drums from Another Planet\\Single Hit\\Kick\\Kick A002.wav"
one_hot = loader.path_to_one_hot(path, 256)

print(one_hot)
print(_create_causal_layer(one_hot).eval(session = sess).shape)
