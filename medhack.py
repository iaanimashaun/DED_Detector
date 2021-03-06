import streamlit as st
import numpy as np
import pandas as pd
import os 
import matplotlib.pyplot as plt 
#import tensorflow as tf 
from tensorflow import keras
#from keras.models import load_model
from tensorflow.keras.preprocessing import image 

st.title('X_ray Image Classifier')

expander = st.sidebar.beta_expander('Want to classify an x-ray?')

expander = st.sidebar.beta_expander('Meet the team')
expander.write('Ibrahim Animashaun (ibrahimanimashaun@gmail.com)')
expander.image('https://wallpapersafari.com/w/0jzAxW')
expander.write('Aderemi Fayoyiwa (aderemifayoyiwa@gmail.com)')
expander.image('/Users/aderemifayoyiwa/Downloads/Aderemi/Aderemi net_9760.jpg')


left_column, right_column = st.beta_columns(2)
pressed = left_column.button('digital image of retina')
if pressed:
	left_column.image('/Users/aderemifayoyiwa/Downloads/retina.jpg')
    #left_column.write('Welcome on board!')

st.header('Introductory Statement')
st.text('This model classifies x-ray images')

st.header('Problem Statement')
st.text('Interpreting x_rays can be a bit tricky...')

st.header('Goal')
st.text('our goal is to ...')

expander = st.beta_expander('Technology used')
expander.write('We used ...')

expander = st.beta_expander('ML Algorithm used and model training')
expander.write('We used ...')

expander = st.beta_expander('EDA')
expander.write('We inspected the class distribution')

#file = st.file_uploader('upload your demo file') 
def file_selector(folder_path='.'): 
	filenames = os.listdir(folder_path) 
	selected_filename = st.selectbox('Select a file', filenames) 
	return os.path.join(folder_path, selected_filename) 
filename = file_selector('/Users/aderemifayoyiwa/Downloads/Ahack') 
st.write('You selected %s' % filename)

#file_selector('/Users/aderemifayoyiwa/Downloads/Aderemi')

def load_mdl():
  model = keras.models.load_model('/Users/aderemifayoyiwa/Downloads/model')
  return model

def predict_class(path):

  #img = plt.imread(file)
  #img = img.resize((224,224))
  img=image.load_img(path, target_size=(224, 224))

  x=image.img_to_array(img)
  x=np.expand_dims(x, axis=0)
  images = np.vstack([x])

  model = load_mdl()

  classes = model.predict(images, batch_size=10)

  predicted_class = ''
  labels = ['Mild', 'Severe', 'Moderate', 'Proliferate_DR', 'No_DR']

  for idx, i in enumerate(classes.squeeze()):
    if i == 1.0:
      predicted_class += labels[idx]
      #print(classes)


  return predicted_class


st.write(predict_class(filename))



