#Importation des librairies

import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
from skimage.transform import resize
import cv2
import urllib.request
import requests
from io import BytesIO

#Chargement des données et du model et création d'une variable 'desired_shape'qui correspond aux dimension des images

data = pd.read_csv('donnée_wedressfair_2.csv')
desired_shape = (100, 100, 3)
model = load_model('model.h5')

#Création d'une fonction permettant de charger une image à partir d'une url ou d'une image en mémoire sous forme de tableau numpy. renvoi none si image pas valide.

def ar_image(image):
    if isinstance(image, str):
        if image.startswith('http://') or image.startswith('https://'):
            # If the image is a URL
            try:
                resp = urllib.request.urlopen(image)
                image_bytes = np.asarray(bytearray(resp.read()), dtype="uint8")
                img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
                return img
            except:
                return None
    
    elif isinstance(image, np.ndarray):
        return image
    else:
        return None

#Création d'une fonction qui prendre une image en entrée et l'utilise pour faire une prédiction à l'aide du modèle. La fonction renvoi le vêtement prédit pour cette image en se basant sur les données du csv.
    
def teste_url(image_s):
    image = ar_image(image_s)
    resized_image = resize(image, desired_shape)
    x_new_processed = np.expand_dims(resized_image, axis=0)
    x_new_processed = x_new_processed.astype('float32') / 255.0

    prediction = model.predict(x_new_processed)

   
    predicted_label = np.argmax(prediction, axis=1)[0]

    predicted_index = data[data['label'] == predicted_label]['Vetement'].index[0]

    predicted_cloth = data.loc[predicted_index, 'Vetement']

    return predicted_cloth

 
#Création d'une fonction "main" qui crée une application Web interactive où l'utilisateur peut saisir l'URL d'une image, effectuer une classification de vêtement à l'aide de la fonction teste_url, puis afficher les informations associées au vêtement prédit.

def main():
    st.title("Application de classification de vêtement")
    st.write("Welcome to my Streamlit app!")
    image_url = st.text_input("Enter Image URL")
    if image_url :
        st.image(image_url, caption=None, width=None, use_column_width=True, clamp=False, channels="RGB", output_format="auto")
    if st.button("Submit"):
        if image_url:            
            result = teste_url(image_url)
            df = data[data['Vetement'] == result]
            df.reset_index(drop=True, inplace=True)
                
            for i in range(0,df.shape[0]):
                col1, col2 = st.columns(2)
                with col1:
                    st.image(df['Image_Source'][i], caption=None, width=None, use_column_width=True, clamp=False, channels="RGB", output_format="auto")

                with col2:
                    st.text(df['Marque'][i])
                    st.text(df['Description'][i])
                    st.text(df['Prix'][i])
                    st.text(df['Taille'][i])
                    st.write("[Link]({})".format(df['Lien'][i]))        

        else:
            st.write("Please enter an image URL.")




if __name__ == '__main__':
    main()


