
# Handwritten digit recognition for MNIST dataset using Convolutional Neural Networks

#Import all required keras libraries
from keras.api.datasets import mnist #Mnist dataset
import keras.api.utils as np_utils #NumPy Utilities
from keras.api.models import load_model #to load saved model


# Step 2: Load and return training and test datasets
def load_dataset():
	# Load dataset X_train, X_test, y_train, y_test via imported keras library
	data = mnist.load_data() #Loads the mnist Data
	(X_train, y_train), (x_test, y_test) = data #assign train and test data
	
	# reshape for X train and test vars - Hint: X_train = X_train.reshape((X_
	# train.shape[0], 28, 28, 1)).astype('float32')
	X_train = X_train.reshape((X_train.shape[0],28,28,1)).astype('float32')
	x_test = x_test.reshape((x_test.shape[0],28,28,1)).astype('float32')

	# 2c. normalize inputs from 0-255 to 0-1 
	X_train = X_train / 255
	x_test = x_test / 255
	#Convert y_train and y_test to categorical classes 
	y_train = np_utils.to_categorical(y_train)
	y_test = np_utils.to_categorical(y_test)
	
	
	return X_train, x_test, y_train, y_test

X_train,x_test,y_train,y_test = load_dataset()
#Load  saved model 
cnn = load_model('digitRecognizer.h5')
# Evaluate model 
loss, accuracy = cnn.evaluate(x_test,y_test, verbose = 0)
print(f'Test loss: {loss}')
print(f'Test accuracy: {accuracy}')

# Code below to make a prediction for a new image.

	
from keras.api.preprocessing.image import load_img
from keras.api.preprocessing.image import img_to_array
from keras.api.models import load_model


#load and normalize new image
def load_new_image(path):
	# oad new image
	newImage = load_img(path, color_mode='grayscale', target_size=(28, 28))
	# Convert image to array
	newImage = img_to_array(newImage)
	# eshape into a single sample with 1 channel 
	newImage = newImage.reshape((1,28,28,1)).astype('float32')
	#normalize image data 
	newImage = newImage / 255
	#return newImage
	return newImage
 
#load a new image and predict its class
def test_model_performance(img_path):
	#Call the above load image function
	img = load_new_image(img_path)
	
 
	#predict the class 
	imageClass = cnn.predict(img)
	#Print prediction result
	print(imageClass[0])
	return imageClass 
	
 
#Test model performance here by calling the above test_model_performance function


img_count = 9
for i in range(1,img_count+1): #iterate through all 9 images.start at 1, img_count +1 to include number 9
	image = f'sample_images/digit{i}.png'
	print(f"Predicting class for image: {image[14:-3]}") #slice image string to only display digit{i}.png
	res = test_model_performance(image) #result of the prediction array

	predicted_class = res[0].argmax() #Get the largest number in prediction array
	print(f"Predicted class for {image[14:-3]}: CLASS: {predicted_class}") #print result
	print() #line break for better readability



