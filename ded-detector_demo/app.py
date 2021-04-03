
import os
import json
import requests
import SessionState
import streamlit as st
import tensorflow as tf
from utils import load_and_prep_image, classes_and_models, update_logger, predict_json
from PIL import Image
import imagehash
import matplotlib.pyplot as plt
#import cv2
from io import BytesIO


# Setup environment credentials 
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'ded-detector-4dcd64335a30.json' # change for your GCP key
PROJECT = "ded-detector" # change for your GCP project
REGION = "us-central1" # change for your GCP region (where your model is hosted)

### Streamlit code ###

st.title("Welcome to DED Detector")
st.header("Detect Diabetic Eye Disease")
#st.image(r'images\7fdb177b8f7d.png')





@st.cache # cache the function so predictions aren't always redone (Streamlit refreshes every click)
def make_prediction(image, model, class_names):
    """
    Takes an image and uses model (a trained TensorFlow model) to make a
    prediction.

    Returns:
     image (preproccessed)
     pred_class (prediction class from class_names)
     pred_conf (model confidence)
    """
 
     
    hash0 = imagehash.average_hash(Image.open(BytesIO(image)) )
    hash1 = imagehash.average_hash(Image.open('retina.png'))
    result = hash0 - hash1

    not_retina = 'It seems you did not upload image of a retina scan'
    if(result < 18):


        image = load_and_prep_image(image)
        # Turn tensors into int16 (saves a lot of space, ML Engine has a limit of 1.5MB per request)
        image = tf.cast(tf.expand_dims(image, axis=0), tf.int16)
        # image = tf.expand_dims(image, axis=0)
        preds = predict_json(project=PROJECT,
                            region=REGION,
                            model=model,
                            instances=image)
        pred_class = class_names[tf.argmax(preds[0])]
        pred_conf = tf.reduce_max(preds[0])
        return image, pred_class, pred_conf
    return image, not_retina, 0

# Pick the model version
choose_model = 'Model'

# Model choice logic
if choose_model == "Model":
    CLASSES = classes_and_models["model_1"]["classes"]
    MODEL = classes_and_models["model_1"]["model_name"]


# Display info about model and classes
if st.checkbox("Show stages of Diabetic Eye Disease"):
    st.write(f"You chose {MODEL}, these are the stages of diabetic eye disease it can identify:\n", CLASSES)

# File uploader allows user to add their own image
uploaded_file = st.file_uploader(label="Upload an image of retina scan",
                                 type=["png", "jpeg", "jpg"])

# Setup session state to remember state of app so refresh isn't always needed
# See: https://discuss.streamlit.io/t/the-button-inside-a-button-seems-to-reset-the-whole-app-why/1051/11 
session_state = SessionState.get(pred_button=False)



expander = st.sidebar.beta_expander('Meet the team')
expander.image('Ibrahim.png')
expander.write('Ibrahim Animashaun')
#expander.markdown('[email](https://mail.google.com/mail/u/0/?fs=1&to=iaanimashaun@gmail.com&su=SUBJECT&body=BODY&tf=cm)')
expander.markdown('[Email](https://mail.google.com/mail/u/0/?fs=1&to=iaanimashaun@gmail.com&su=SUBJECT&body=BODY&tf=cm)    [GitHub](https://github.com/iaanimashaun)   [Linkedin](https://www.linkedin.com/in/iaanimashaun)')


expander.write()
expander.image('Aderemi.png')
expander.write('Aderemi Fayoyiwa')
#expander.markdown('[Email](https://mail.google.com/mail/u/0/?fs=1&to=aderemifayoyiwa@gmail.com&su=SUBJECT&body=BODY&tf=cm)')
expander.markdown('[Email](https://mail.google.com/mail/u/0/?fs=1&to=aderemifayoyiwa@gmail.com&su=SUBJECT&body=BODY&tf=cm)   [GitHub](https://github.com/AderemiF)   [Linkedin](https://www.linkedin.com/in/aderemi-fayoyiwa)')




# Create logic for app flow
if not uploaded_file:
    st.warning("Please upload an image.")
    st.stop()
else:
    session_state.uploaded_image = uploaded_file.read()
    st.image(session_state.uploaded_image, use_column_width=True)
    pred_button = st.button("Predict")

# Did the user press the predict button?
if pred_button:
    session_state.pred_button = True 

# And if they did...
if session_state.pred_button:
    session_state.image, session_state.pred_class, session_state.pred_conf = make_prediction(session_state.uploaded_image, model=MODEL, class_names=CLASSES)
    if session_state.pred_conf != 0:
        st.write(f"Prediction: {session_state.pred_class}, \
               Confidence: {session_state.pred_conf:.3f}")
    else:
        st.write(f"Prediction: {session_state.pred_class}")


            

# TODO: code could be cleaned up to work with a main() function...
# if __name__ == "__main__":
#     main()

