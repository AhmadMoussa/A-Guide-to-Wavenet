import tensorflow as tf
import matplotlib.pyplot as plt
from audio_loader_and_encoder import lqe

loader = lqe()

path = "D:\\Users\\amm65\\Desktop\\Audio Sample Dataset\\Shroom_Drums from Another Planet\\Shroom_Drums from Another Planet\\Single Hit\\Kick\\Kick A001.wav"
sample_rate = 11025
duration = 0.1
mono = True
audio = loader.load_audio(path, sample_rate, duration, mono)

plt.plot(audio)
plt.show()

quantization_channels = 256

sess = tf.InteractiveSession()
encoded = loader.mu_law_encode(audio, quantization_channels)
encoded_data = encoded.eval(session = sess)

plt.plot(encoded_data)
plt.show()

one_hot = loader.audio_to_one_hot(encoded, 1, quantization_channels)
one_hot_data = one_hot.eval(session = sess)
print(one_hot_data)

decoded = loader.mu_law_decode(encoded, quantization_channels)
decoded_data = decoded.eval(session = sess)
plt.plot(decoded_data)
plt.show()
