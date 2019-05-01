# Wavenet
A guide to understanding what a Wavenet is

## Introduction:
Wavenet was first introduced in this [paper](https://arxiv.org/abs/1609.03499) and is an essential read for the rest of this article. 

## Terms we need to understand:
I found that reading research papers I would come across a lot of words and terms that I couldn't understand, and they were not explained as it is assumed that you have some knowledge in the field that is being discussed. But if you've just started then a lot of the terms will be a difficult to digest. There will be sections throughout this article that will breka down the important ideas.
Tractable: you will come across this term in the context of solving problems, computing/calculating things and creating models. In the context of solving a problem, when we state that a problem is "tractable", it means that it can be solved or that the value can be found in a reasonable amount of time. Some problems are said to be "Intractable". From a computational complexity stance, intractable problems are problems for which there exist no efficient algorithms to solve them. Most intractable problems have an algorithm – the same algorithm – that provides a solution, and that algorithm is the brute-force search.
Latent: a variable is said to be latent, if it can't be observed directly but has to be "inferred" somehow
Stochastic: something that was randomly determined

## Recurrent Neural Networks(RNN):
* [Introduction to neural networks by kaparthy](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) Wonderful Article to start with, along with great examples and code on github
* [Code for the above mentione article](https://github.com/karpathy/char-rnn)
* How can we make neural networks more exciting? We make them accept sequences as input rather than having a fixed number of inputs.
* 

## Convolutional Network:
In mathematics the word convolution refers to "a mathematical operation on two functions to produce a third function that expresses how the shape of one is modified by the other" [from wikipedia](https://en.wikipedia.org/wiki/Convolution). We will see the similarity to this in a second. In machine learning, a convolution is an activation layer that returns a feature map. The way it does that, is rather interesting.
Sometimes it is beneficial to look at the surroundings of a given spot (neuron) and focus on a smaller area, rather than the entire data given to us. This area of interest will be called the "receptive field". The lens with which we inspect this receptive field in our neural network will be called a "Filter". The filter is but a small layer of parameters. Why is it called a filter? Because we are going to place this filter over our area of interest and pull information through it to learn something new about our data. After that we slide our filter to a new area of interest and repeat.
Ultimately we end up with an output layer, called an activation map, or feature map. Which gives us a vague idea of what we learned from the convolutional process. If we keep repeating this process we can gain a deeper understanding of our initial input.
[Convolutional Layer](http://cs231n.github.io/convolutional-networks/#conv): a layer over which a filter is being applied
[Dilated Convolution](https://www.quora.com/What-is-the-difference-between-dilated-convolution-and-convolution+stride): In this type of convolution, the filter expands. This is sometimes also called "Atrous" convolution, where "Atrous" comes from the french word "à trous" meaning "with holes".
[Causal Concolution](https://arxiv.org/abs/1803.01271): this term is rather vague, but a "causal" convolution means that there is no information leakage from future to past.
Fractional Upsampling: when the stride of the filter is less than 1 (S < 1). Then we end up a with a feature map size larger than the input layer size.
