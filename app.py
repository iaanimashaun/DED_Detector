import streamlit as st
import numpy as np
import pandas as pd
import os 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow import keras
import base64
#from keras.models import load_model
from tensorflow.keras.preprocessing import image 

main_bg = '/Users/aderemifayoyiwa/Downloads/back.jpg'
main_bg_ext = 'jpg'

st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
    }}
    </style>
    """,
    unsafe_allow_html=True
)



st.title('RED - Detector')

expander = st.sidebar.beta_expander('Want to see an x-ray classified?')
expander.write('In a short while')

expander = st.sidebar.beta_expander('Meet the team')
expander.image('/Users/aderemifayoyiwa/Downloads/Ibrahim.png')
expander.write('Ibrahim Animashaun \niaanimashaun@gmail.com')
expander.image('/Users/aderemifayoyiwa/Downloads/Aderemi/Aderemi_net_9760.jpg')
expander.write('Aderemi Fayoyiwa \naderemifayoyiwa@gmail.com')



left_column, right_column = st.beta_columns(2)
pressed = left_column.button('Digital image of retina')
if pressed:
	left_column.image('/Users/aderemifayoyiwa/Downloads/retina.jpg')
    #left_column.write('Welcome on board!')

st.header('Introduction')
st.text('Diabetes is becoming a global epidemic with the number of people affected worldwide \nrising from 108 million in 1980 to an estimated 425 million in 2017, \nand an estimated 629 million in 2045.')
st.text('Diabetic retinopathy is a complication of diabetes. \nIt is caused by high blood sugar levels damaging the back of the eye (retina). \nIt can cause blindness if left undiagnosed and untreated.\nThe prevalence of diabetic retinopathy (DR), a primary cause of blindness \nand vision loss worldwide, was estimated at 93 million in 2012 \nout of which 28 million people had vision-threatening DR, \nthis is also expected to further increase.')
st.text('Diabetic eye disease is a leading cause for blindness registration among \nworking age adults in England and Wales. \nDiabetic retinopathy accounts for 4% of an estimated 1.93 million people living \nwith sight loss in the UK. \nWith the increasing prevalence of diabetes it is also of public health importance to \naddress diabetic eye screening')

st.header('Problem Statement')
st.text('Increased rate of diabetes and ageing population poses significant challenge \nto performing diabetic retinopathy (DR) screening for patients. \nThese screenings are mostly based on the analysis of fundus photographs by specialist.\nThe cost of such systems can put substantial strain on the healthcare systems when both \nfinancial and human resources are often in short supply and scaling and sustaining \nsuch systems has been found to be challenging.')

st.header('Goal')
st.text('Our goal for this project is: \nTo train an algorithm that can detect abnormal (unhealthy) retinal images. \nTo help increase access to screenings for diabetic retinopathy and \nother eye diseases. \nTo help reduce the cost while improving screening accuracy. \nPatients can visit a health screening centre that takes a picture of the eyes and tells \nyou when you need to see a doctor.')

expander = st.beta_expander('Technologies used')
expander.write('Github')
expander.write('Streamlit')
expander.write('Heroku')

expander = st.beta_expander('ML Algorithm used and model training')
expander.write('Model: CNN(EfficientNetB0)')


expander = st.beta_expander('Data')
expander.write('Source: https://www.kaggle.com/c/aptos2019-blindness-detection/overview')
expander.write('Big Data: 20GB')
expander.write('Sample subset')
expander.write('Train: 2931 images, Validation: 731 images (20%)')

expander = st.beta_expander('Preprocessing')
expander.write('Gaussian filter')
expander.image('/Users/aderemifayoyiwa/Downloads/gaussianfilter.png')
expander.write('Resizing - (224, 224)')
expander.write('Transformations')


expander = st.beta_expander('EDA')
expander.write('We inspected the class distribution')

expander = st.beta_expander('Image classification')
expander.image('/Users/aderemifayoyiwa/Downloads/piechart.png')

expander = st.beta_expander('Performance')
expander.write('Validation accuracy: 74%')
expander.image('/Users/aderemifayoyiwa/Downloads/accuracy.png')

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



