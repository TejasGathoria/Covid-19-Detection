from flask import Flask, render_template, request
from flask_uploads import UploadSet, configure_uploads, IMAGES
from keras.preprocessing.image import img_to_array
#from scipy.misc import imsave, imread, imresize
import numpy as np
import re
import sys
import os
import cv2
import torch
import torchvision

#initalize our flask app
app = Flask(__name__)

global model,class_names
#initialize these variables
class_names = ['Normal', 'Viral', 'Covid']

model = torch.load('weights/resnet18_model.pth')
model.eval()
photos = UploadSet('photos', IMAGES)

app.config['UPLOADED_PHOTOS_DEST'] = '.'
configure_uploads(app, photos)

@app.route('/')
def index():
	#initModel()
	#render out pre-built HTML file right on the index page
	return render_template("index.html")


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    
	if request.method == 'POST' and 'photo' in request.files:
		filename = photos.save(request.files['photo'])
		#os.rename('./'+filename,'./'+'output.png')

	img = cv2.imread(filename)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = cv2.resize(img, (224,224))
	transform = torchvision.transforms.Compose([
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])

	img = transform(img)
	img = img.unsqueeze(0)
	print(img.shape)
	output = model(img)
	_, preds = torch.max(output, 1)
	softmax = torch.nn.Softmax(dim=1)
	prob = softmax(output)[0]
	print(prob)
	print(class_names[preds])
	os.remove(filename)
	return render_template("index2.html", s1 = class_names[0], s2= prob[0].item()*100, s3= class_names[1], s4=prob[1].item()*100, s5=class_names[2], s6=prob[2].item()*100)
 
if __name__ == "__main__":
	#decide what port to run the app in
	#os.remove('output.png')
	port = int(os.environ.get('PORT', 5000))
	#run the app locally on the givn port
	app.run(host='0.0.0.0', port=port)
	#optional if we want to run in debugging mode
	#app.run(debug=True)