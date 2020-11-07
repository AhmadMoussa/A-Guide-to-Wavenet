# Wavenet (Work in Progress)
A compendium of everything you need to know to get started with Wavenet. From creating the basic building blcoks to a complete Wavenet Model as well as feeding and training your model on some data and ultimately generating your own sounds. 

# Prerequisites:
* Wavenet was first introduced in the paper [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499) and is an essential read for the rest of this article. Don't worry if there are parts that are not clear after reading the paper, that's what this guide is for!
* What makes the wavenet so powerful? As we're going through the different parts that will ultimately conglomerate into this network, we will notice that a lot of ideas have been borrowed from other types of networks, and all of them are powerful mechanisms in their own right, such as convolutional layers, dilated filters, gated activation units, 1by1 convolutions for channel shrinking, skip connections and residual connections. But they also work towards fixing some of the problems that previous deep networks struggled with in the past.

# Tools:
* python
* tensorflow & keras (Mainly keras because it helps keeping things simple)

# Relevant Articles:
* [A paper a day delays the neuron decay](https://towardsdatascience.com/residual-blocks-building-blocks-of-resnet-fd90ca15d6ec)
* [Korean Guy](https://medium.com/@kion.kim/wavenet-a-network-good-to-know-7caaae735435)

# The Shape of Sound:
## [What are Sampling Rate and Bit Depth?](https://www.presonus.com/learn/technical-articles/sample-rate-and-bit-depth) 
* To understand the underlying inner workings of the wavenet, we need to first take a closer look at the data that we are going to use. Because without Data in the first place, there would be no need for this neural network anyway. We need to train our model on audio. Easier said than done. First and foremost we need to find a way to convert from audio, that we humans perceive and "understand" with our ears, to a format that is machine understandable (Spoiler: Numbers!). 


* Sound, as we hear it in the real world, can be thought of as a continuous analog waveform (continuous vibrations in the air). Converting this analog waveform to a number representation is done by capturing it's descriptive values at successive points in time. analogically it's somewhat like capturing a video, which is ultimately just a succesion of images. Later on, we can chain these descriptive values (samples) together and to accurately recreate the original waveform. Naturally, the more "snapshot" we take of a given sound the better we will be able to recreate it later on with a good "resolution". Hence, the rate of capture is called the "sampling Rate". Bit Depth stands for the number of bits that are used to represent each captured sample. (talk about what bit depth does and help quantize signal-to-noise ratio (SNR))

## [From Audio to Time Series](https://en.wikipedia.org/wiki/Time_series)
* Now, we can think of the audio data points that we captured as a time series, which is simply said, a bunch of data points that have some correlation and causality with each other in relation to time.
* But we have a little-not-so-little problem. There is a massive amount of these sample data points, as well as a gigantic dynamic range for each sound.


#### Why do these two factors cause us problems?
* At the end of the day, we're going to want to generate some audio samples. Our nework is going to try to recreate the sample data points that we recorded and fed into our machine. Assume these audio files are Encoded in Stereo 16-bit. That means that there are 65536 values along the y-axis where data points could be located. Now we are not going to delve into madness and try to assume that our network is going to take an educated guess as to where it should place the data point at a given time step t. Luckily, there is a way to reduce this humonguous number of values to a smaller range, specifically 256. Now that's a number that I can work with! 

## [μ-law Quantization or Companding Transformation](http://digitalsoundandmusic.com/5-3-8-algorithms-for-audio-companding-and-compression/)
* An ingenious way to shrink our dynamic range. To understand what we are actually doing we should primarily have a look at a visual analogy. 

![An illustration of the Weber–Fechner law. On each side, the lower square contains 10 more dots than the upper one. However the perception is different: On the left side, the difference between upper and lower square is clearly visible. On the right side, the two squares look almost the same.](https://i.imgur.com/84bBwCm.png)

* So, what exactly are we looking at? Observe the two squares to the left side. Which one of them has more dots? Obviously, the lower one. Now observe the two squares to the right. Guess what, it's a little bit more difficult to say which one has more dots now (It's still the lower one, the lower squares have exactly 10 dots more than their upper counterparts). This is known as the Weber-Fechner law, which treats the relation between the actual change in a physical stimulus and the perceived change in the stimulus. 
* What does this have to do with Audio? Speech, for example, has a very high dynamic range, and when we record someone speaking, we want to capture the big frequency jumps that their voice makes, and compared to these jumps, subtle variations and finer details are lost in comparision. Hence with the mu-law companding transformation we can compress a waveform and represent it with significantly less bits, without loosing any important information of the original data. And here's the formula:

![mu-law](https://wikimedia.org/api/rest_v1/media/math/render/svg/2df208f7dd18fc678447dbffac60b8ca21eaffba)

Looks scary, but it's not really.

* I found several implementations [korean guy](https://github.com/AhmadMoussa/WaveNet-gluon/blob/master/utils.py) and [lemonzi](https://github.com/ibab/tensorflow-wavenet/blob/master/test/test_mu_law.py) but somehow I couldn't get either to work "right", I'll have to look into it at some later point.
* A quick python implementatio of the mu-law encode, we can use numpy or torch (couldn't figure it out yet though):
```python
# function that will create the mu-law encoding from the input waveform vector
def encode_mu_law(to_encode, mu = 256):
    mu = mu -1
    toplog = np.log(1 + (np.abs(to_encode) * mu))
    botlog = np.log(1 + mu)
    sign = np.sign(to_encode)
    fx =  sign * ( toplog / botlog )
    return ((fx + 1) / 2 * mu + 0.5).astype(np.long)
```
But we still have to convert to the desired range that we want to project onto, namely -256,256

# Model Structure:

## Causal Dilated Convolutions:

### Convolutions:
* Let's digress a bit and start easy, if you don't know what a convolution is I recommend you go for a little stroll, and read this wonderfully comprehensive [beginner's guide by Adit Deshpande](https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks/) (this is the most comprehensive and beginner friendly read I could find)

* Sometimes it is beneficial to look at the surroundings of a given spot (neuron) and focus on a smaller area, rather than the entire data given to us. We can learn a whole lot by observing the relationship between some data and it's surrounding data. In the case of neural networks that deal with images, it is not practical to use a fully connected feedforward neural network (Even though we can still learn features with this architecture). This concise area of interest that we are going to inspect in detail, is usually called a "receptive field". The tool (we can also refer to it as a lens) with which we inspect this receptive field is reffered to as a "Filter". 

* What does the filter look like? The filter is but a small layer of parameters, simply said, a weight matrix. Why is it called a filter? Because we are going to place this filter over our area of interest (figuratively) and pull information through it (dot products between the entries of the filter and the input at any position, or rather element wise multiplications) to learn something new about our data. After that we slide our filter to a new area of interest and repeat. Performing this sliding movement around the picture can also be reffered to as convolving, wherefrom stems the term convolutional neural network (CNN).

![Convolutional Layer](https://i.imgur.com/IxrbGAg.png)

* You must wonder: what is this filter actually doing? In theory we are taking some numbers from the input layer and multiplying them with the weights in the filter, to get some new numbers to describe what is happening in the picture with some abstraction. Ultimately, we end up with an output layer, called an activation map, or feature map. This activation map, represents what the network thinks, is in the image. If we keep repeating this process our model gains a deeper understanding of the initial input.

* This was a brief introduction to convolutional layers. If you are hungry for more convolutional neural network shenanigans, I suggest you read this [course by Stanford University on COnvolutional Neural Networks](http://cs231n.github.io/convolutional-networks/#conv)

And here's the code for a simple 2D convolution that detects vertical lines using keras:
```python
from numpy import asarray
from keras import Sequential
from keras.layers import Conv2D

# define input data
data = [[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0]]

# Creating a numpy array from the data above
data = asarray(data)

'''
    Here we are converting our data to a 4Dimensional container
    Think of it as an array of tensors (Tensor being a 3Dimensional Array)
    Such that [number of samples, columns, rows, channels]
    In this trivial case we only have one sample, and the channels are shallow
'''
data = data.reshape(1, 8, 8, 1)
print(data)

# Create a Sequential keras model, we'll only have one layer here
model = Sequential()
# https://keras.io/layers/convolutional/
# Conv2D(number of filters, tuple specifying the dimension of the convolution window, input_shape)
model.add(Conv2D(1, (3,3), input_shape=(8, 8, 1)))

# Define a vertical line detector
detector = [[[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]]]
weights = [asarray(detector), np.asarray([0.0])]
# store the weights in the model
model.set_weights(weights)
# confirm they were stored
print(model.get_weights())

# apply filter to input data
yhat = model.predict(data)

for r in range(yhat.shape[1]):
	# print each column in the row
	print([yhat[0,r,c,0] for c in range(yhat.shape[2])])

```

this [tutorial](https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/) helped with this example.

___

### [Dilations](https://www.quora.com/What-is-the-difference-between-dilated-convolution-and-convolution+stride):
* Now let's expand the concept of a filter (literally and figuratively). 

![Standard Convolution](https://i.imgur.com/WweMRuM.gif) ![Dilated Convolution](https://i.imgur.com/InbYG23.gif)

* What are dilated convolutions? A dilated convolution refers to a convolution with a filter that is dilated, where dilated means that the filter is enlarged by spacing out the weight values and padding them with zeroes in between. What is really happening, is that we are expanding the receptive field and increasing our coverage, we are looking at the relationship between neighbours that are a little bit more distant from each other. This is useful if some important features of our data are only definable in regions larger than what our receptive field covers. To define such a feature, normally one would have to convovle as second time, or use a dilated filter. Below is an illustrative schematic of such a filter:

![Dilated Filter](https://i.imgur.com/MnoFWNn.png)

* [Why are dilated convolutions useful?](https://stackoverflow.com/questions/41178576/whats-the-use-of-dilated-convolutions)

`Stacked dilated convolutions enable networks to have very large receptive fields with just a few layers, while preserving the input resolution throughout the network as well as computational efficiency.`

* Let's dissect this,beginning with "resolution". Usually, when the filter convolves over an input, we end up with an activation map that has a lesser size than what we started with. Imagine it being like a funnel being applied to each area of the input. In that sense, we are losing resolution. The more convolutional layers we have, the more our input will shrink. `One approach involves repeated up-convolutions that aim to recover lost resolution while carrying over the global perspective from downsampled layers (Noh et al., 2015; Fischer et al., 2015). This leaves open the question of whether severe intermediate downsampling was truly necessary` (Chen et al., 2015; Yu & Koltun, 2016).

![Normal Convolution](https://i.imgur.com/RnkTsA6.gif)

* Hence in the case of audio, preserving resolution and ordering of data is one of our main priorities.

[Original Paper on Dilated Convolutions](https://arxiv.org/pdf/1511.07122.pdf)

### Causality:
* Talk about difference between causal convolution and RNN and how causal convolution is easier to compute. Add how dilating the filter fixes the problem that the causal convolution has.

#### [What's a Causal Convolution?](https://www.quora.com/What-are-causal-convolutions)
* Causality in our case simply means that we are hiding all future time steps(data samples), from the filter that is currently at work. We don't want it to see yet what's coming up in the future, but we would like it to learn the relationship between the current time-step and previous ones. If there would be "leakage" of future content into the current time-step, then we wouldn't be teaching our network the correct content. The origin of this causal layer can be drawn back, yet again, to the pixelCNN where instead of causality they use masks, but it is basically the same. 

* In the PixelCNN, in simple terms, we are trying to learn what the next pixel should look like (it's RGB values), given all the pixels that have occurred previously. Think about it intuitively for a second, we've been given a half complete image that is mostly blue-ish of color, it would make sense to complete the sequence with another blue pixel, maybe a little bit brighter or less bright.

* And this is the reason why the authors have labelled PixelCNN as a model that performs "Autoregressive Density Estimation", where density stands for a [probability density function](https://en.wikipedia.org/wiki/Probability_density_function). In both Wavenet and PixelCNN we are trying to model the joint probability of a timestep as a product of conditional probabilities of all previous timesteps. In layman's terms, what's the chance we're simething is going to happen, given everything that has happened already.

![Probability Density](https://i.imgur.com/28Xnr0P.png)

#### [Why use Causal Convolutions rather than RNNs or LSTMs](https://arxiv.org/pdf/1803.01271.pdf)
* Recurrent Neural Networks are notorious for being hard to train(vanishing/exploding gradient and recurrent connections), but they have an advantage, which is a long memory retention. 

* Now our causal filter has to be linearly very large to achieve an effective history size, which is cumbersome to compute and introduces an overhead. But this problem can simply be solved by stacking dilated layers on top of this causal one, which allows us to exponentially increase our filter size and achieve a larger history size. 

* In addition, the backpropagation method is different from that used in RNNs, and therefore avoiding the vanishing/exploding gradient problem altogether.


## Gated Activation Units:
* The term "Gate" has been adopted in a number of fields, for example in music production, we use the term "Noise-Gate" when we refer to a device that is responsible for attenuating signals that fall below some pre determined threshhold, and simpler said, if a certain sound is not loud enough, then the listener will not be able to hear it at all. 

* What are activation functions and why are they useful? As it's name indicates, it can be viewed as a mechanism that decides, based on the amplitude of an incoming signal, if it should send a signal or not. Usually, the type of the problem we are dealing with, determines which type of activation functions would be best suited. Let's have a look at three different types of these functions:

I think I should write a tutorial for these activation functions separately, as it blows up the range of this article quite a bit.
#### Sigmoid:

![Sigmoid Function](https://i.imgur.com/iQF327F.png)

Historically, this has been the most widely used activation function (before it was replaced by the ReLU). Why is the sigmoid function used? And what makes it a good activation function? It has several nice properties. One of them is differentiability, this is especially important when the gradient needs to be calculated during back-propagation. It is also non-linear, this allows us to stack layers.

![Linear function](https://i.imgur.com/0h3TtqR.png)

A stack of linear functions would be equivalent to have a single one. Therefore with this 

#### tanh:

#### Rectified Linear Unit:


![Gated Activation Unit Function](https://i.imgur.com/CtVxRtC.png)

* Apparently these gates yield better results than using a rectified linear unit. Pin-pointing why they do so 

* Using this type of gate, it allows us to control what information will be propagated throughout the remaining layers. This concept has been adopted from LSTM models where gates are employed to establish a "long term memory" for important information, as opposed to RNNs which struggle retaining such information. Dauphin et. al. offers a good explanation for this in section 3 of his paper on "Language Modeling with Gated Convolutional Networks":


> LSTMs enable long-term memory via a separate cell controlled by input and forget gates. This allows information to flow unimpeded through potentially many timesteps. Without these gates, information could easily vanish through the transformations of each timestep.
	In contrast, convolutional networks do not suffer from the same kind of vanishing gradient and we find experimentally that they do not require forget gates. Therefore, we consider models possessing solely output gates, which allow the network to control what information should be propagated through the hierarchy of layers.
	
And here's the [paper](https://arxiv.org/pdf/1612.08083.pdf) for reference.

* Another reason why this concept was adopted might be because PixelRNN have been outperforming PixelCNN due to them having recurrent connections between one layer to all other layers in the network, whereas PixelCNN does not have this feature. This is solved by adding more layers to add depth, and since CNNs don't struggle with vanishing gradients we don't have to worry about that, whereas RNNs do. We also add a gate to mimic the gates of the inner workings of LSTMs and have more control over the flow of information through our model.

Additionally, one last point I would like to add, is that generally sigmoid and tanh are better for approximating classifier functions. Hence this might lead to faster training and convergence. Therefore, this could also be a reason, why they chose to replace the ReLU with thei custom function.

## 1 x 1 Convolution:

* This seems like a really trivial thing, when it is applied in a 2 Dimensional context, then we would simply be multiplying each number in a matrix by another number.

* But in higher dimensions this allows to do a rather non-trivial computation on the input volume. Assume we have a tensor of dimensions 6 x 6 x 32 (each channels has 32 attributes). Now, if we'd multiply this tensor by a 1 x 1 x 32 filter, we would end up with a output that has 6 x 6 x 1 dimensionality. Pretty cool, huh? With a simple 1 x 1 filter we were able to shrink the number of channels. This idea was first proposed in [this paper](https://arxiv.org/pdf/1312.4400.pdf).

This effectively allows us to shrink the number of channels to the number of filters that were applied.

![1 x 1 convolution](https://i.imgur.com/OSNKo6j.png)

## Residual Blocks:
* The term "Residual Block" refers the structure that computes the residual, which will be passed on to subsequent residual blocks. What is a residual

# Implementing the network:

## Dependencies:

* We'll need a couple of things before we get started:
`pip install tensorflow`
`pip install tensorboard`
`pip install librosa`
`pip install numpy`
`pip install matplotlib`

## Before we start:
* Make sure you install librosa and tensorflow. Librosa might throw some errors, hence try debugging it first and getting it to work. we will need it to load audio samples into our model.

```python
import librosa
import numpy as np
import matplotlib.pyplot as plt

def load_audio():
    filename = librosa.util.example_audio_file()
    audio, _ = librosa.load(filename, 11025, mono=True, duration = 0.1)
    audio = audio.reshape(-1, 1)
    return audio

plt.plot(load_audio())
plt.show()
```

* Let's create a minimal audio loader, and test it by plotting a small part of the waveform. You should get something that looks like this:

![audio graph](https://i.imgur.com/8IUIsKH.png)

* Next step is quantizing this audio by passing it through the mu-law encode:

```python
def mu_law_encode(audio, quantization_channels):
    mu = tf.to_float(quantization_channels - 1)
    safe_audio_abs = tf.minimum(tf.abs(audio), 1.0)
    magnitude = tf.log1p(mu * safe_audio_abs) / tf.log1p(mu)
    signal = tf.sign(audio) * magnitude
    return tf.to_int32((signal + 1) / 2 * mu + 0.5)

quantized_audio = mu_law_encode(audio, 256)

sess = tf.InteractiveSession()
print(quantized_audio.shape)

for i in range(quantized_audio.shape[0]):
    print(quantized_audio[i].eval(session = sess))
```
![Quantized audio graph](https://i.imgur.com/PYb7avL.png)

* Looking good so far. Notice the scale of the two graphs on their y-axes, even though it seems like the quantized graph has a much larger scale, it actually only spans 256 discrete values, whereas the prior graph has an almost continuous decimal range. The first graph has a better resolution visually, but the quantized version is good enough to work with. Also, notice the second huge dip, it is almost vertical in the quantized version as aopposed to the original. This reinforces that our encode pays more detail to the little variations more than it does for the huge frequency swings.

* Now last thing before we can move one is converting this quantized audio data into one hot encodings.

```python
def _one_hot(input_batch):
    encoded = tf.one_hot(input_batch, depth = 256, dtype = tf.float32)
    shape = [1, -1, 256]
    encoded = tf.reshape(encoded, shape)
    return encoded

one_hot = _one_hot(quantized_audio)
one_hot_data = one_hot[0,0,:].eval(session=sess)
print(one_hot_data)
```
* And you should see something like this:

```
[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
```

* The only value that is not zero, is the 128th, which is in concordance with what we see in the graphs and when we print out the content of the arrays. 

## Enter Wavenet:

* Our input data has to go through a causal convolution first, the purpose of it is to shrink the number of layers. What this is in fact, is a feature pooling technique as discussed previously in 1x1 convoltuion section.

```python
import tensorflow as tf

def create_variable(name, shape):
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    variable = tf.Variable(initializer(shape=shape), name=name)
    return variable

def _create_causal_layer(input_batch):
        initial_filter_width = 32
        initial_channels = 2**8
        residual_channels = 16
        weights_filter = create_variable('filter', [initial_filter_width, initial_channels, residual_channels])
        return causal_conv(input_batch, weights_filter, 1)

# what this essentially does is reduce the number of channels and
def causal_conv(value, filter_, dilation, name='causal_conv'):
    filter_width = tf.shape(filter_)[0]
    restored = tf.nn.conv1d(value, filter_, stride=1, padding='VALID')
    tf.global_variables_initializer().run()
    # Remove excess elements at the end.
    out_width = tf.shape(value)[1] - (filter_width - 1) * dilation
    result = tf.slice(restored, [0, 0, 0], [-1, out_width, -1])
    return result
```
We are also cutting a couple of elements at the end depending on the size of the batch that was fed to it.

## Structure and Helper Functions:

* Let's first create all the parts that will make up the final network, wou'll need to understand [variable scopes](https://stackoverflow.com/questions/35919020/whats-the-difference-of-name-scope-and-a-variable-scope-in-tensorflow) and the [python `with` statement](https://preshing.com/20110920/the-python-with-statement-by-example/). Go ahead and look them up and come back:

* We need afunction that creates variables according to our specified shape, and we'll have a number of different ones. So let's slap a wrapper around the tensorflow `Variable()` function:
```
import tensorflow as tf

def create_variable(name, shape):
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    variable = tf.Variable(initializer(shape=shape), name=name)
    return variable
```

* Next let's create the structure of the network, which is basically a large dictionary of learnable filters that we're gonna use to convolve over the input sequence, and update them during backprop. Code:

```python
import tensorflow as tf
from create_variable import create_variable

def construct_network(filter_width, quantization_channels,residual_channels,dilation_channels,skip_channels,dilations):
    net = dict()

    with tf.variable_scope('wavenet'):
        with tf.variable_scope('causal_layer'):
            layer = dict()
            layer['filter'] = create_variable('filter', [filter_width, quantization_channels,residual_channels])
            net['causal_layer'] = layer
        net['dilated_stack'] = list()
        with tf.variable_scope('dilated_stack'):
            for i, dilation in enumerate(dilations):
                with tf.variable_scope('layer{}'.format(i)):
                    current = dict()
                    current['filter'] = create_variable('filter', [filter_width, residual_channels, dilation_channels])
                    current['gate'] = create_variable('gate', [filter_width, residual_channels, dilation_channels])
                    current['dense'] = create_variable('dense',[1, dilation_channels,  residual_channels])
                    current['skip'] = create_variable('skip', [1, dilation_channels, skip_channels])
                    net['dilated_stack'].append(current)
        with tf.variable_scope('postprocessing'):
            current = dict()
            current['postprocess1'] = create_variable('postprocess1',[1, skip_channels, skip_channels])
            current['postprocess2'] = create_variable('postprocess2',[1, skip_channels, quantization_channels])
        net['postprocessing'] = current
    return net
```

* Essentially, what we're doing here, is creating a massive python dictionary that will point towards "containers", that describe and carry our layers. This dictionary is comprised of: 

	1. A python dictionary that contains the causal layer at the front of the network.
	2. A python list of dictionaries that represents the dilated stack, wherein each dict is representing a residual block with a filter, gate, dense and skip layer. 
	3. And finally a dictionary 2 postprocessing layers at the end.
	
Here's a diagram to help your imagination:
![data structure](https://i.imgur.com/gxZdPx8.png)

* Now let's define some useful operations:
	1. `causal_conv` we're going to use this one a bunch. It has two modes, a simple 1D convolution that serves as a channel shrinking operation when the dilation factor is equal to 1. And another mode when the dilation factor is higher than 1. For which we'll need the other two functions, `time_to_batch` and `batch_to_time`.
	2. `time_to_batch`
	3. `batch_to_time`

```python
def time_to_batch(value, dilation, name = None):
	with tf.name_scope('time_to_batch'):
		shape = tf.shape(value)
		pad_elements = dilation - 1 - (shape[1] + dilation - 1) % dilation
		padded = tf.pad(value, [[0, 0], [0, pad_elements], [0, 0]])
		reshaped = tf.reshape(padded, [-1, dilation, shape[2]])
		transposed = tf.transpose(reshaped, perm = [1, 0, 2])
		return tf.reshape(transposed, [shape[0] * dilation, -1, shape[2]])
		
def batch_to_time(value, dilation, name = None):
	with tf.name_scope('batch_to_time'):
		shape = tf.shape(value)
		prepared = tf.reshape(value, [dilation, -1, shape[2]])
		transposed = tf.transpose(prepared, perm = [1, 0, 2])
		return tf.reshape(transposed, [tf.div(shape[0], dilation), -1, shape[2]])
		
def causal_conv(value, filter_, dilation, name = 'causal_conv'):
	with tf.name_scope(name):
		filter_width = tf.shape(filter_)[0]
		if dilation > 1:
			transformed = time_to_batch(value, dilation)
			conv = tf.nn.conv1d(transformed, filter_, stride = 1, padding = 'VALID')
			restored = batch_to_time(conv, dilation)
		else:
			restored = tf.nn.conv1d(value, filter_, stride = 1, padding = 'VALID')
			
			out_width = tf.shape(value)[1] - (filter_width - 1) * dilation
			result = tf.slice(restored, [0,0,0], [-1, out_width, -1])
			return result

```
* The convolutions we are going to use are all causal. Hence we're creating a function that takes our input `value`, the `filter` which we are going to convolve with over the input, and the dilation parameter to specify how dilated the filter is. If the dilation argument is less or equal to 1 we just convolve normally. If it's larger than 1 we need to some processing on the input. The input is in the form of a one_hot encoding

## Terms we need to understand:
I found that while reading research papers, I would come across a lot of words and terms that I couldn't understand, and they were not explained as it is assumed that you have some knowledge in the field that is being discussed. But if you've just started then a lot of the terms will be a difficult to digest. There will be sections throughout this article that will breka down the important ideas.
* Tractable: you will come across this term in the context of solving problems, computing/calculating things and creating models. In the context of solving a problem, when we state that a problem is "tractable", it means that it can be solved or that the value can be found in a reasonable amount of time. Some problems are said to be "Intractable". From a computational complexity stance, intractable problems are problems for which there exist no efficient algorithms to solve them. Most intractable problems have an algorithm – the same algorithm – that provides a solution, and that algorithm is the brute-force search.
* Latent: a variable is said to be latent, if it can't be observed directly but has to be "inferred" somehow
Stochastic: something that was randomly determined

## Recurrent Neural Networks(RNN):
* [Introduction to neural networks by kaparthy](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) Wonderful Article to start with, along with great examples and code on github
* [Code for the above mentione article](https://github.com/karpathy/char-rnn)
* How can we make neural networks more exciting? We make them accept sequences as input rather than having a fixed number of inputs.

## Convolutional Network:

* [Convolutional Layer](http://cs231n.github.io/convolutional-networks/#conv): a layer over which a filter is being applied
* [Dilated Convolution](https://www.quora.com/What-is-the-difference-between-dilated-convolution-and-convolution+stride): In this type of convolution, the filter expands. This is sometimes also called "Atrous" convolution, where "Atrous" comes from the french word "à trous" meaning "with holes".
* [Causal Concolution](https://arxiv.org/abs/1803.01271): this term is rather vague, but a "causal" convolution means that there is no information leakage from future to past.
* Fractional Upsampling: when the stride of the filter is less than 1 (S < 1). Then we end up a with a feature map size larger than the input layer size.

## Problems with librosa:
* I ran into an error using librosa, which wouldn't allow it to load audio files or read them. This github issue accurately describes this probem and offers solutions for different environments [`raise NoBackendError()`](https://github.com/librosa/librosa/issues/219). On windows however, [this](https://github.com/adaptlearning/adapt_authoring/wiki/Installing-FFmpeg) fixed it for me. Also, you'll need this for windows:
	1. to open admin command prompt press Windows+X and select the admin prompt.
	2. If it shows powershell instead of command prompt got to settings->personalize->change to command prompt
