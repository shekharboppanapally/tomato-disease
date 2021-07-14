import streamlit as st
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pickle

st.title('TOMATO DISEASE PREDICTION AND PREVENTION APP')
st.write('This is a tomato disease prediction web app using streamlit.')
st.text('upload an image')
model=pickle.load(open('tomato.pkl','rb'))

uploaded_file=st.file_uploader('choose an image',type='jpg')
if uploaded_file is not None:
  img=imread(uploaded_file)
  st.image(img,caption='uploaded image')
  if st.button('PREDICT'):
    CATAGORIES=['tomato_early_blight','tomato_healthy','tomato_late_blight']
    st.write('Results')
    flat_data=[]
    img=np.array(img)
    img_resized=resize(img,(150,150,3))
    flat_data.append(img_resized.flatten())
    flat_data=np.array(flat_data)
    y_out=model.predict(flat_data)
    q=model.predict_proba(flat_data)
    for index,item in enumerate(CATAGORIES):
       st.write(f'{item} : {q[0][index]*100}')
    y_out=CATAGORIES[y_out[0]]   
    if y_out=='tomato_early_blight':
      st.title('DISEASE:TOMATO EARLY BLIGHT')
      st.subheader('HOW TO PREVENT TOMATO EARLY BLIGHT?')
      st.write('Tomatoes that have early blight require immediate attention before the disease takes over the plants. Thoroughly spray the plant (bottoms of leaves also) with Bonide Liquid Copper Fungicide concentrate or Bonide Tomato & Vegetable. Both of these treatments are organic. If possible time applications so that 12 hours of dry weather follows applications. A day after treatment, remove the lower branches with sharp razor blade knife. Clean your knife with rubbing alcohol before trimming the next plant to prevent the spread of the disease. Repeat fungicide treatments every 7 to 14 days. Read the label instructions carefully. Do not spray pesticides, fungicides, fertilizers or herbicides when it’s in the high 80’s or 90; you can damage your plants. Water your plants the day before spraying, hydration is important!')
    elif y_out=='tomato_healthy':
      st.title('DISEASE:NO DISEASE')
      st.subheader('your tomato plant is healthy')
    else:
      st.title('DISEASE:TOMATO LATE BLIGHT')
      st.subheader('HOW TO PREVENT TOMATO LATE BLIGHT?')
      st.write('Sanitation is the first step in controlling tomato late blight. Clean up all debris and fallen fruit from the garden area. This is particularly essential in warmer areas where extended freezing is unlikely and the late blight tomato disease may overwinter in the fallen fruit.plants should be inspected at least twice a week. Since late blight symptoms are more likely to occur during wet conditions, more care should be taken during those times.For the home gardener, fungicides that contain maneb, mancozeb, chlorothanolil, or fixed copper can help protect plants from late tomato blight. Repeated applications are necessary throughout the growing season as the disease can strike at any time. For organic gardeners, there are some fixed copper products approved for use; otherwise, all infected plants must be immediately removed and destroyed.')    
st.write('HOW TO USE')
st.write('step1.first click on the option browse files,it will open the camera and files in your mobile')
st.write('step2.choose any of the option camera or files and take a picture of your tamoto plant')
st.write('step3.after loading the image,click on predict button then it will predict that your tomato plant is diseased or not,if diseased it predicts the disease and inform how to cure it')
st.write('It is made by using Transfer Learning technique')
st.write("Transfer Learning is the reuse of a pre trained model on a new problem.It's currently very popular in deep learning because it can train deep neural network with comparatively little data.This is very useful in data science field since most of real world problems do not have millions of labelled data")
st.write('Accuracy of this model is :85%')
st.write('For more queries please contact:shekharboppanapally944@gmail.com')    