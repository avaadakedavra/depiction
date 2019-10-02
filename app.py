#our web app framework!

#you could also generate a skeleton from scratch via
#http://flask-appbuilder.readthedocs.io/en/latest/installation.html

#Generating HTML from within Python is not fun, and actually pretty cumbersome because you have to do the
#HTML escaping on your own to keep the application secure. Because of that Flask configures the Jinja2 template engine 
#for you automatically.
#requests are objects that flask handles (get set post, etc)
from flask import Flask, render_template,request
#scientific computing library for saving, reading, and resizing images
from scipy.misc import imsave, imread, imresize
#for matrix math 
import numpy as np
#for importing our keras model
import keras.models
#for regular expressions, saves time dealing with string data
import re
import base64
 
#system level operations (like loading files)
import sys 
#for reading operating system data
import os
#tell our app where our saved model is
sys.path.append(os.path.abspath("./model"))
#from load import * 
from Capsules import *
#initalize our flask app
app = Flask(__name__)
#global vars for easy reusability
global model, graph
import pickle as pkl
#initialize these variables
#model, graph = init()

#decoding an image from base64 into raw representation

def convertImage(imgData1):
	imgstr = re.search(b'base64,(.*)',imgData1).group(1)
	with open('output.png','wb') as output:
		output.write(base64.b64decode(imgstr))

'''
	def generate_image_from_text(self, text):
    noise = np.zeros(shape=(1, self.random_input_dim))
    encoded_text = np.zeros(shape=(1, self.text_input_dim))
    encoded_text[0, :] = self.glove_model.encode_doc(text)
    noise[0, :] = np.random.uniform(-1, 1, self.random_input_dim)
    generated_images = self.generator.predict([noise, encoded_text], verbose=0)
    generated_image = generated_images[0]
    generated_image = generated_image * 127.5 + 127.5
    return Image.fromarray(generated_image.astype(np.uint8))	
'''
def img_from_normalized_img(normalized_img):
    image = normalized_img * 127.5 + 127.5
    return Image.fromarray(image.astype(np.uint8))


@app.route('/')
def index():
	#initModel()
	#render out pre-built HTML file right on the index page

	return render_template("index.html")

@app.route('/output',methods=['GET','POST'])
def output():



	#current_dir = 'Documents/Flask/venv1/app/ImageClassificationGANs'
	current_dir = os.getcwd()
	#print(current_dir)
	model_dir_path = os.path.join(os.getcwd(),'model')

	#print(model_dir_path)

	inputText = request.get_data() 
	#print(inputText)
	#type(inputText)


	text = inputText.decode("utf-8")
	
	#str(inputText,'utf-8')
	#str(inputText,'unicode')

	#inputText = text 
	#print(text)
	#print(type(text))
	#inputText = "Hello this is a string"
	filename = os.path.join(os.getcwd(),'results_array.pickle')
	fileObject2 = open(filename, 'rb')
	image_label_pairs = pkl.load(fileObject2)
	fileObject2.close()




	#inputText = text 
	gans = Capsules()
	gans.load_model(model_dir_path)
	
	test_dir = os.path.join(os.getcwd(), 'static')
	print(test_dir)

	np.random.shuffle(image_label_pairs)

	for i in range(3):
		image_label_pair = image_label_pairs[i]
		normalized_image = image_label_pair[0]
		text = image_label_pair[1]
		print(text)
		
		image = img_from_normalized_img(normalized_image)
		#image.save(os.path.join(current_dir, Capsules.model_name + '-generated-' + str(i) + '-0.jpg'))
		for j in range(3):
			#image.save(os.path.join(test_dir, Capsules.model_name + '-generated-' + str(i) + '-' + str(j) + '.jpg'))
			generated_image = gans.generate_image_from_text(text)
			generated_image.save(os.path.join(test_dir, Capsules.model_name + '-generated-' + str(i) + '-' + str(j) + '-' + '.jpg'))
    	
	#inputText = '/static'+ '/' + Capsules.model_name+ '-generated-2-2.jpg'
	#print(inputText)

	'''
	for j in range(3):
		generated_image = gans.generate_image_from_text(inputText)
		generated_image.save(os.path.join(test_dir, Capsules.model_name + '-generated-' + str(j) + '.jpg'))
	'''
	return inputText
	#return inputText

@app.route('/predict',methods=['GET','POST'])
def predict():
	#whenever the predict method is called, we're going
	#to input the user drawn character as an image into the model
	#perform inference, and return the classification
	#get the raw data format of the image
	imgData = request.get_data()
	#encode it into a suitable format
	convertImage(imgData)
	#print ("debug")
	#read the image into memory
	x = imread('output.png',mode='L')
	#compute a bit-wise inversion so black becomes white and vice versa
	x = np.invert(x)
	#make it the right size
	x = imresize(x,(28,28))
	#imshow(x)
	#convert to a 4D tensor to feed into our model
	x = x.reshape(1,28,28,1)
	#print ("debug2")
	#in our computation graph
	with graph.as_default():
		#perform the prediction
		out = model.predict(x)
		print(out)
		print(np.argmax(out,axis=1))
		#print ("debug3")
		#convert the response to a string
		response = np.array_str(np.argmax(out,axis=1))
		return response	
	
		
if __name__ == "__main__":
	#decide what port to run the app in
	port = int(os.environ.get('PORT', 5000))
	#run the app locally on the givn port
	app.run(host='127.0.0.1', port=port)
	#optional if we want to run in debugging mode
	#app.run(debug=True)