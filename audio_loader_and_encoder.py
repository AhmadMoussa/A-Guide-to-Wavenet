import librosa
import tensorflow as tf

class lqe():
    def load_example(self):
        filename = librosa.util.example_audio_file()
        audio, _ = librosa.load(filename, 11025, mono=True, duration = 0.1)
        audio = audio.reshape(-1, 1)
        return audio

    def load_audio(self, path, sample_rate, duration, mono):
        audio, _ = librosa.load(path, sample_rate, mono, duration)
        return audio.reshape(-1,1)

    def mu_law_encode(self, audio, quantization_channels):
        mu = tf.cast(quantization_channels - 1, dtype = tf.float32)
        safe_audio_abs = tf.minimum(tf.abs(audio), 1.0)
        magnitude = tf.log1p(mu * safe_audio_abs) / tf.log1p(mu)
        signal = tf.sign(audio) * magnitude
        return tf.cast((signal + 1) / 2 * mu + 0.5, tf.int32)

    def mu_law_decode(self, output, quantization_channels):
        mu = quantization_channels - 1
        signal = 2 * (tf.to_float(output) / mu) - 1
        magnitude = (1 / mu) * ((1 + mu)**abs(signal) - 1)
        return tf.sign(signal) * magnitude

    def audio_to_one_hot(self, input_batch, batch_size, quantization_channels):
        return tf.reshape(tf.one_hot(input_batch, quantization_channels, dtype = tf.float32), [batch_size, -1, quantization_channels])

    def path_to_one_hot(self, path, quantization_channels):
        return self.audio_to_one_hot(self.mu_law_encode(self.load_audio(path, 11025, 0.1, True), quantization_channels), 1, quantization_channels)
