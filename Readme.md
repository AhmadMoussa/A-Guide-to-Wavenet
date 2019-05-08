# Wavenet
A compendium of everything you need to know to get started with Wavenet. From turning Audio into Data, Creating the Wavenet Model, feeding and training your model on your data and ultimately generate your own sounds. (Hopefully :p)

## Introduction:
* Wavenet was first introduced in this [paper](https://arxiv.org/abs/1609.03499) and is an essential read for the rest of this article. 
* Why is it so good?

## Tools:
* python
* pytorch
* theano

## Relevant Articles:
* [A paper a day delays the neuron decay](https://towardsdatascience.com/residual-blocks-building-blocks-of-resnet-fd90ca15d6ec)
* [Korean Guy](https://medium.com/@kion.kim/wavenet-a-network-good-to-know-7caaae735435)

## From Audio to Data:
### [What are Sampling Rate and Bit Depth?](https://www.presonus.com/learn/technical-articles/sample-rate-and-bit-depth) 
* To understand the underlying inner workings of the wavenet, we need to first take a closer look at the data that we are going to use. Because without Data in the first place, there would be no need for this neural network anyway. We need to train our model on audio. Easier said than done. First and foremost we need to find a way to convert from audio, that we humans perceive and "understand" with our ears, to a format that is machine understandable (Spoiler: Numbers!). 


* Sound, as we hear it in the real world, can be thought of as a continuous analog waveform (continuous vibrations in the air). Converting this analog waveform to a number representation is done by capturing it's descriptive values at successive points in time. analogically it's somewhat like capturing a video, which is ultimately just a succesion of images. Later on, we can chain these descriptive values (samples) together and to accurately recreate the original waveform. Naturally, the more "snapshot" we take of a given sound the better we will be able to recreate it later on with a good "resolution". Hence, the rate of capture is called the "sampling Rate". Bit Depth stands for the number of bits that are used to represent each captured sample. (talk about what bit depth does and help quantize signal-to-noise ratio (SNR))

### [From Audio to Time Series](https://en.wikipedia.org/wiki/Time_series)
* Now, we can think of the audio data points that we captured as a time series, which is simply said, a bunch of data points that have some correlation and causality with each other in relation to time.
* But we have a little-not-so-little problem. There is a massive amount of these sample data points, as well as a gigantic dynamic range for each sound.


##### Why do these two factors cause us problems?
* At the end of the day, we're going to want to generate some audio samples. Our nework is going to try to recreate the sample data points that we recorded and fed into our machine. Assume these audio files are Encoded in Stereo 16-bit. That means that there are 65536 values along the y-axis where data points could be located. Now we are not going to delve into madness and try to assume that our network is going to take an educated guess as to where it should place the data point at a given time step t. Luckily, there is a way to reduce this humonguous number of values to a smaller range, specifically 256. Now that's a number that I can work with! 

### [μ-law quantization or companding transform](http://digitalsoundandmusic.com/5-3-8-algorithms-for-audio-companding-and-compression/)
* An ingenious way to shrink our dynamic range. To understand what we are actually doing we should primarily have a look at a visual analogy. 

![An illustration of the Weber–Fechner law. On each side, the lower square contains 10 more dots than the upper one. However the perception is different: On the left side, the difference between upper and lower square is clearly visible. On the right side, the two squares look almost the same.](https://i.imgur.com/84bBwCm.png)

* So, what exactly are we looking at? Observe the two squares to the left side. Which one of them has more dots? Obviously, the lower one. Now observe the two squares to the right. Guess what, it's a little bit more difficult to say which one has more dots now (It's still the lower one, the lower squares have exactly 10 dots more than their upper counterparts). This is known as the Weber-Fechner law, which treats the relation between the actual change in a physical stimulus and the perceived change in the stimulus. 
* What does this have to do with Audio? Speech, for example, has a very high dynamic range, and when we record someone speaking, we want to capture the big frequency jumps that their voice makes, and compared to these jumps, subtle variations and finer details are lost in comparision. Hence with the mu-law algorithm we can compress a waveform and represent it with significantly less bits, without loosing any important information of the original data.
* I found several implementations [korean guy](https://github.com/AhmadMoussa/WaveNet-gluon/blob/master/utils.py) and [lemonzi](https://github.com/ibab/tensorflow-wavenet/blob/master/test/test_mu_law.py) but somehow I couldn't get either to work "right", I'll have to look into it at some later point.
* A quick python implementatio of the mu-law encode, we can use numpy or torch (couldn't figure it out yet though):
``` 
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

## Model Structure:

### Causal Dilated Convolutions

#### Convolutions:
* Let's digress a bit and start easy, if you don't know what a convolution is I recommend you go for a little stroll, and read this wonderfully comprehensive [beginner's guide by Adit Deshpande](https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks/) (this is the most comprehensive and beginner friendly read I could find)

* Sometimes it is beneficial to look at the surroundings of a given spot (neuron) and focus on a smaller area, rather than the entire data given to us. We can learn a whole lot by observing the relationship between some data and it's surrounding data. In the case of neural networks that deal with images, it is not practical to use a fully connected feedforward neural network (Even though we can still learn features with this architecture). This concise area of interest that we are going to inspect in detail, is usually called a "receptive field". The tool (we can also refer to it as a lens) with which we inspect this receptive field is reffered to as a "Filter". 

* What does the filter look like? The filter is but a small layer of parameters, simply said, a weight matrix. Why is it called a filter? Because we are going to place this filter over our area of interest (figuratively) and pull information through it (dot products between the entries of the filter and the input at any position, or rather element wise multiplications) to learn something new about our data. After that we slide our filter to a new area of interest and repeat. Performing this sliding movement around the picture can also be reffered to as convolving, wherefrom stems the term convolutional neural network (CNN).

* ![Convolutional Layer](https://i.imgur.com/IxrbGAg.png)

* You must wonder: what is this filter actually doing? In theory we are taking some numbers from the input layer and multiplying them with the weights in the filter, to get some new numbers to describe what is happening in the picture with some abstraction. Ultimately, we end up with an output layer, called an activation map, or feature map. This activation map, represents what the network thinks, is in the image. If we keep repeating this process our model gains a deeper understanding of the initial input.

* This was a brief introduction to convolutional layers. If you are hungry for more convolutional neural network shenanigans, I suggest you read this [course by Stanford University on COnvolutional Neural Networks](http://cs231n.github.io/convolutional-networks/#conv)

```

'''
    x: DxCxHxW
    D -> number of inputs
    C -> number of image channels
    H -> image height
    W -> width of image

    W: NFxCxHFxHW
    NF -> number of the filter in the filter map
    C -> number of channels (should be same as channels in the picture)
    HF -> height of the filter
    HW -> is the width of the filer

    b: is our bias
'''

def conv_forward(X, W, b, stride = 1, padding = 1):
    pass

```

[convolutional layer code](https://wiseodd.github.io/techblog/2016/07/16/convnet-conv-layer/)


#### Dilations:
* Now let's expand the concept of a filter (literally and figuratively). 
* ![Standard Convolution](https://i.imgur.com/WweMRuM.gif) ![Dilated Convolution](https://i.imgur.com/InbYG23.gif)
* What are dilated convolutions? A dilated convolution refers to a convolution with a filter that is dilated, where dilated means that the filter is enlarged by spacing out the weight values and padding them with zeroes in between. Below is an illustrative schematic:
* ![Dilated Filter](https://i.imgur.com/MnoFWNn.png)

#### Causality:
* Talk about difference between causal convolution and RNN and how causal convolution is easier to compute. Add how dilating the filter fixes the problem that the causal convolution has.

## Terms we need to understand:
I found that reading research papers I would come across a lot of words and terms that I couldn't understand, and they were not explained as it is assumed that you have some knowledge in the field that is being discussed. But if you've just started then a lot of the terms will be a difficult to digest. There will be sections throughout this article that will breka down the important ideas.
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
