from flask import Flask, redirect, url_for, render_template, request
import os 
import tensorflow as tf
import cv2 as cv
#from cv2 import cv2 as cv
import numpy as np


app = Flask(__name__, static_url_path='/static')

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

def load_mdl():
  model = tf.keras.models.load_model('model')
  return model

def predict_class(image):
  
    #img = image.load_img(path, target_size =(224,224))

    #x = tf.image.resize_with_pad(img, 224, 224, method='nearest', antialias=True)
    #  x = keras.preprocessing.image.img_to_array(x)
    #  x = np.expand_dims(x, axis=0)
    #  x = np.vstack([x])


    # Needed if you use OpenCV, By default, it use BGR instead RGB
    img = cv.cvtColor(np.float32(image), cv.COLOR_BGR2RGB)

    # Resize image to match with input model
    cv.resize(img, (224, 224))

    # Convert to Tensor of type float32 for example
    image_tensor = tf.convert_to_tensor(img, dtype=tf.float32)

    # Add dimension to match with input mode 
    image_tensor = tf.expand_dims(image_tensor, 0)


    #x = image.img_to_array(img)
    #x = np.expand_dims(x, axis=0)

    #images = np.vstack([x])
    model = load_mdl()
  
    classes = model.predict(image_tensor) #, batch_size=10

    predicted_class = ''
    labels = ['Mild', 'Severe', 'Moderate', 'Proliferate_DR', 'No_DR']

    for idx, i in enumerate(classes.squeeze()):
        if i == 1.0:
            predicted_class += labels[idx]
            #print(classes)

    return predicted_class

@app.route('/')
def home():
    return render_template('index.html', content="Testing")



@app.route('/model/')
def model():
    return render_template('model.html')
    

@app.route('/contact/')
def contact():
    return render_template('contact.html')


@app.route('/demo/')
def demo():
    return render_template('demo.html')




@app.route('/upload/', methods=['POST'])
def upload():
    target = os.path.join(APP_ROOT, 'images/')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist('file'):
        #print(file)
        filename = file.filename
        destination = '/'.join([target, filename])
        #print(destination)
        file.save(destination)
        img = cv.imread(os.path.join(destination))
        pred = predict_class(img)
       

    #return render_template('classify.html')
    return ('<h1>pred</h1>')




if __name__ == '__main__':
    app.run(port=4555, debug=True)






# Needed if you use OpenCV, By default, it use BGR instead RGB
#image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

# Resize image to match with input model
#image = cv.resize(image, (32, 32))

# Convert to Tensor of type float32 for example
#image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)

# Add dimension to match with input mode 
#image_tensor = tf.expand_dims(image_tensor, 0)

# After you can predict image for example :
#predictions = probability_model.predict(
#        image_tensor, use_multiprocessing=True)