
# Handwritten digit recognition for MNIST dataset using Convolutional Neural Networks


#----------------Using Keras 3.3.2 with recent version of tensorflow------------------------
import keras

#NOTE: These were the only way for me to access/import the needed libraries for Kera... not too sure why
from keras.api.datasets import mnist #Mnist dataset
import keras.api.utils as np_utils #NumPy Utilities

#import matplotlib.pyplot as plt; used to graph results



#Load and return training and test datasets
#Preprocessing step
def load_dataset():
	# Load dataset X_train, X_test, y_train, y_test via imported keras library
	data = mnist.load_data() #Loads the mnist Data
	(X_train, y_train), (x_test, y_test) = data #assign train and test data
	
	
	# reshape for X train and test vars 
	X_train = X_train.reshape((X_train.shape[0],28,28,1)).astype('float32')
	x_test = x_test.reshape((x_test.shape[0],28,28,1)).astype('float32')

	# normalize inputs from 0-255 to 0-1 - 
	X_train = X_train / 255
	x_test = x_test / 255
	# Convert y_train and y_test to categorical classes 
	y_train = np_utils.to_categorical(y_train)
	y_test = np_utils.to_categorical(y_test)
	# return your X_train, X_test, y_train, y_test

	
	return X_train, x_test, y_train, y_test


def digit_recognition_cnn():
	#create CNN model here with Conv + ReLU + Flatten + Dense layers
	cnn = keras.Sequential()
	#Convolution2D + RELU
	cnn.add(keras.layers.Conv2D(filters=30, kernel_size=(5,5),activation='relu'))
	#MaxPooling 2D
	cnn.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))
	#Convolution 2D + RELU
	cnn.add(keras.layers.Conv2D(filters=15, kernel_size=(3,3),activation='relu'))
	#MaxPooling 2D
	cnn.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))
	#Dropout with 20% probability
	cnn.add(keras.layers.Dropout(rate=0.2))
	#Flatten
	cnn.add(keras.layers.Flatten())
	#Dense layers
	#cnn.add(keras.layers.Dense(units=128, activation='relu'))
	cnn.add(keras.layers.Dense(units=200, activation='relu')) #more units added, Provided better results
	cnn.add(keras.layers.Dense(units=50, activation='relu'))
	cnn.add(keras.layers.Dense(units=10, activation='softmax'))
	# Compile model
	cnn.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics=['accuracy'])
	# return
	return cnn

# Call digit_recognition_cnn() to build model
model = digit_recognition_cnn()

X_train, x_test, y_train, y_test = load_dataset() #load data
# Set epochs to a number between 10 - 20 and batch_size between 150 - 200
history = model.fit(X_train,y_train,epochs=10,batch_size=150, validation_data=(x_test,y_test))

#Model Evaluation
loss, accuracy = model.evaluate(x_test,y_test)
print(f'Test loss: {loss}')
print(f'Test accuracy: {accuracy}')

#Save Model
model.save('digitRecognizer.h5')

''' GRAPH STUFF
###############################################################################
# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
###########################################################################
'''

