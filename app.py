import streamlit as st
import numpy as np
import pandas as pd
import os 
import matplotlib.pyplot as plt 
#import tensorflow as tf 
#from tensorflow import keras
import base64
#from keras.models import load_model
#from tensorflow.keras.preprocessing import image 





st.markdown(' # **DED - Detector**')
st.image('./images/retina.png')
expander = st.sidebar.beta_expander('Want to try it out?')
expander.markdown('[click here for demo](https://ded-detector.uc.r.appspot.com)')

expander = st.sidebar.beta_expander('Awards')
expander.markdown('[MedHack 2021](https://devpost.com/software/red-detector)')




expander = st.sidebar.beta_expander('Meet the team')
expander.image('./images/Ibrahim.png')
expander.write('Ibrahim Animashaun')
#expander.markdown('[email](https://mail.google.com/mail/u/0/?fs=1&to=iaanimashaun@gmail.com&su=SUBJECT&body=BODY&tf=cm)')
expander.markdown('[Email](https://mail.google.com/mail/u/0/?fs=1&to=iaanimashaun@gmail.com&su=SUBJECT&body=BODY&tf=cm)    [GitHub](https://github.com/iaanimashaun)   [Linkedin](https://www.linkedin.com/in/iaanimashaun)')


expander.write()
expander.image('images/Aderemi_net_9760.png')
expander.write('Aderemi Fayoyiwa')
#expander.markdown('[Email](https://mail.google.com/mail/u/0/?fs=1&to=aderemifayoyiwa@gmail.com&su=SUBJECT&body=BODY&tf=cm)')
expander.markdown('[Email](https://mail.google.com/mail/u/0/?fs=1&to=aderemifayoyiwa@gmail.com&su=SUBJECT&body=BODY&tf=cm)   [GitHub](https://github.com/AderemiF)   [Linkedin](https://www.linkedin.com/in/aderemi-fayoyiwa)')

 



left_column, right_column = st.beta_columns(2)
pressed = left_column.button('Digital image of retina')
#if pressed:
#	left_column.image('images\retina.jpg')
    #left_column.write('Welcome on board!')

st.markdown('# Introduction')
st.markdown(' ### Diabetes is becoming a global epidemic with the number of people affected worldwide rising from 108 million in 1980 to an estimated 425 million in 2017, and an estimated 629 million in 2045.')
st.markdown(' ##### \nDiabetic retinopathy is a complication of diabetes. \nIt is caused by high blood sugar levels damaging the back of the eye (retina). \nIt can cause blindness if left undiagnosed and untreated.\nThe prevalence of diabetic retinopathy (DR), a primary cause of blindness \nand vision loss worldwide, was estimated at 93 million in 2012 \nout of which 28 million people had vision-threatening DR, \nthis is also expected to further increase.')
st.markdown(' ##### \nDiabetic eye disease is a leading cause for blindness registration among \nworking age adults in England and Wales. \nDiabetic retinopathy accounts for 4% of an estimated 1.93 million people living \nwith sight loss in the UK. \nWith the increasing prevalence of diabetes it is also of public health importance to \naddress diabetic eye screening')

st.markdown(' # Problem Statement')
st.markdown('##### \n * Increased rate of diabetes and ageing population poses significant challenge \nto performing diabetic retinopathy (DR) screening for patients. \n * These screenings are mostly based on the analysis of fundus photographs by specialist.\n * The cost of such systems can put substantial strain on the healthcare systems when both \nfinancial and human resources are often in short supply and scaling and sustaining \nsuch systems has been found to be challenging.')

st.markdown('# Goal')
st.markdown(' ##### \nOur goal for this project is: \n * To train an algorithm that can detect abnormal (unhealthy) retinal images. \n * To help increase access to screenings for diabetic retinopathy and \nother eye diseases. \n * To help reduce the cost while improving screening accuracy. \n * Patients can visit a health screening centre that takes a picture of the eyes and tells \nyou when you need to see a doctor.')

st.markdown(' # Exploratory Data Analysis')
expander = st.beta_expander('Data')
expander.write('Source: https://www.kaggle.com/c/aptos2019-blindness-detection/overview')
expander.write('Big Data: 20GB')
expander.write('Sample subset')
expander.write('Train data: 2931 images, Validation data: 731 images (20%)')


expander = st.beta_expander('Preprocessing')
expander.write('Gaussian filter')
expander.write('Resizing - (224, 224) and other transformations')
expander.write('~500MB')

expander = st.beta_expander('Plots')
expander.image('images/piechart.png')
expander.image('images/barchart.png')





#expander = st.beta_expander('Image classification')
st.markdown('# Modelling')
expander = st.beta_expander('ML Algorithm used and model training')
expander.write('Transfer Learning')
expander.write('Model: CNN(EfficientNetB0)')
expander.write('Validation accuracy: 74%')
expander.image('images/accuracy.png')




expander = st.beta_expander('Technologies Used')
expander.write('Python')
expander.write('Google Colab')
expander.write('Github')
expander.write('Streamlit')
expander.write('Heroku')
expander.write('Google Cloud Platform')



st.markdown('# Challenges')
expander = st.beta_expander('Limitations')
expander.write('Computation power')
expander.write('Time')
expander.write('Data')
expander.write('Class imbalance')




expander = st.beta_expander('What next?')
expander.write('Identify more pathologies')
expander.write('Scalalable')
expander.write('Classification + Localisation')

#file = st.file_uploader('upload your demo file') 
def file_selector(folder_path='.'): 
	filenames = os.listdir(folder_path) 
	selected_filename = st.selectbox('Select a file', filenames) 
	return os.path.join(folder_path, selected_filename) 


#file_selector('/Users/aderemifayoyiwa/Downloads/Aderemi')
@st.cache(suppress_st_warning=True)
def load_mdl():
  model = tf.keras.models.load_model('model')
  return model

def predict_class(path):

  img = image.load_img(path, target_size =(224,224))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)

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


#filename = file_selector('./test_files/') 
#st.write('You selected %s' % filename[13:])
#pred = predict_class(filename)
#st.write(' ## The predicted class is %s' % pred)



