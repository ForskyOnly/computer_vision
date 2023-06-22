from ultralytics import YOLO
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw

# Charger le modèle
def load_model():
    model = YOLO("best.pt")
    return model

# Créer une classe pour transformer le flux vidéo
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = load_model()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Appliquer le modèle
        preds = self.model.predict(source=img, conf=0.5, iou=0.5, save=False)

        # Dessiner les résultats sur l'image
        for i, result in enumerate(preds):
            for j, box in enumerate(result.boxes.data):
                cv2.rectangle(img,
                              (int(box[0]), int(box[1])),
                              (int(box[0] + box[2]), int(box[1] + box[3])),
                              (255, 0, 0),
                              2)
                cv2.putText(img, 
                            f"Classe : {result.names[int(box[5])]}, Confiance : {box[4]}", 
                            (int(box[0]), int(box[1]-10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.9, 
                            (36,255,12), 
                            2)

        return img

# Dessiner les prédictions sur l'image
def draw_preds(image, preds):
    draw = ImageDraw.Draw(image)

    for i, result in enumerate(preds):
        for j, box in enumerate(result.boxes.data):
            # Convertir les coordonnées des boîtes
            x1, y1 = int(box[0]), int(box[1])
            x2, y2 = int(box[0] + box[2]), int(box[1] + box[3])

            # Dessiner le rectangle
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)

            # Ajouter le texte
            text = f"Classe : {result.names[int(box[5])]}, Confiance : {box[4]}"
            draw.text((x1, y1-10), text, fill=(36,255,12))

    return image

# Streamlit app
def app():
    model = load_model()
    st.header("YOLO Real Time Detection")
    st.sidebar.title("Settings")
    confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)
    option = st.sidebar.selectbox("Choose option", ("Image", "Camera"))
    
    if option == "Image":
        st.header("Image Processing")
        uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image.', use_column_width=True)

            rotate_direction = st.sidebar.radio("Rotation", ["Original", "90 degrees", "-90 degrees"])
            if rotate_direction == "90 degrees":
                image = image.rotate(-90)
            elif rotate_direction == "-90 degrees":
                image = image.rotate(90)
            st.image(image, caption='Rotated Image.', use_column_width=True)

            if st.button("Predict"):
                frame = np.array(image)

                # Effectuer les prédictions sur l'image pivotée
                preds = model.predict(source=frame, conf=confidence_threshold, iou=0.5, save=False)

                # Dessiner les prédictions sur l'image
                image = draw_preds(image, preds)

                # Afficher l'image avec les prédictions
                st.image(image, caption='Predicted Image.', use_column_width=True)

                # Afficher les prédictions
                for i, result in enumerate(preds):
                    st.write(f"Résultat {i+1} :")
                    for j, box in enumerate(result.boxes.data):
                        st.write(f"Boîte {j+1} :")
                        st.write(f"Position (x, y) : ({box[0]}, {box[1]})")
                        st.write(f"Largeur : {box[2]}")
                        st.write(f"Hauteur : {box[3]}")
                        st.write(f"Confiance : {box[4]}")
                        st.write(f"Classe : {result.names[int(box[5])]})")

    elif option == "Camera":
        st.header("Real Time Detection")
        webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

# Run the app
app()
