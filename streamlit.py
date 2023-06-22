from ultralytics import YOLO
import streamlit as st
import cv2
import numpy as np
from PIL import Image


@st.cache(allow_output_mutation=True)
def load_model():
    model = YOLO("best.pt")
    return model

model = load_model()

confidence_threshold = st.sidebar.slider("Seuil de confiance", 0.0, 1.0, 0.5, 0.01)

uploaded_file = st.file_uploader("Télécharger une image", type=['png', 'jpg', 'jpeg'])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Image téléchargée.', use_column_width=True)

if st.button("Activer la caméra"):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        st.image(frame, channels="BGR", caption="Image capturée depuis la caméra.")

        preds = model.predict(source=frame, conf=confidence_threshold, iou=0.5, save=False)

        for i, result in enumerate(preds):
            st.write(f"Résultat {i+1} :")
            for j, box in enumerate(result.boxes.data):
                st.write(f"Boîte {j+1} :")
                st.write(f"Position (x, y) : ({box[0]}, {box[1]})")
                st.write(f"Largeur : {box[2]}")
                st.write(f"Hauteur : {box[3]}")
                st.write(f"Confiance : {box[4]}")
                st.write(f"Classe : {result.names[int(box[5])]}")

        if not st.button("Arrêter la caméra"):
            break

    cap.release()
