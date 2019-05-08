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
