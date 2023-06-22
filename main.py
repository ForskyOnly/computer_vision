# Importer les bibliothèques nécessaires
from ultralytics import YOLO
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw

def load_model():
    """
    Charger le modèle YOLO à partir du fichier 'best.pt'.
    
    Returns:
    model: Objet YOLO prêt à effectuer des prédictions.
    """
    model = YOLO("best.pt")
    return model

# Créer une classe pour transformer le flux vidéo
class VideoTransformer(VideoTransformerBase):
    def __init__(self, confidence_threshold, overlap_threshold):
        """
        Initialiser le VideoTransformer avec le modèle, un seuil de confiance et un seuil de recouvrement.
        
        Args:
        confidence_threshold (float): Seuil de confiance pour la prédiction.
        overlap_threshold (float): Seuil de recouvrement pour la suppression non-maximale.
        """
        self.model = load_model()
        self.confidence_threshold = confidence_threshold
        self.overlap_threshold = overlap_threshold

    def transform(self, frame):
        """
        Transformer un frame de la vidéo en détectant les objets à l'aide du modèle YOLO et en dessinant les résultats.

        Args:
        frame (np.ndarray): Frame de la vidéo à analyser.

        Returns:
        img (np.ndarray): Image avec les prédictions dessinées.
        """
        img = frame.to_ndarray(format="bgr24")

        # Appliquer le modèle
        preds = self.model.predict(source=img, conf=self.confidence_threshold, iou=self.overlap_threshold, save=False)

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
                            (36, 255, 12), 
                            2)

        return img

def draw_preds(image, preds):
    """
    Dessiner les prédictions sur l'image.

    Args:
    image (PIL.Image): Image sur laquelle dessiner les prédictions.
    preds (list): Liste de prédictions à dessiner.

    Returns:
    image (PIL.Image): Image avec les prédictions dessinées.
    """
    draw = ImageDraw.Draw(image)

    for i, result in enumerate(preds):
        for j, box in enumerate(result.boxes.data):
            # Convertion des coordonnées des boites
            x1, y1 = int(box[0]), int(box[1])
            x2, y2 = int(box[0] + box[2]), int(box[1] + box[3])
            
            # Dessine le rectangle
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
            text = f"Classe : {result.names[int(box[5])]}, Confiance : {box[4]}"
            
            # Ajout du texte
            draw.text((x1, y1-10), text, fill=(36, 255, 12))

    return image

def app():
    """
    Fonction principale pour l'application Streamlit. Gère l'interaction avec l'utilisateur et affiche les résultats.
    """
    st.title("Alphabet du langage des signes")
    st.sidebar.title("Réglages")

    model = load_model()

    # Paramètres conf et iou
    confidence_threshold = st.sidebar.slider("Confidence Threshold:", 0.0, 1.0, 0.5, 0.01)
    overlap_threshold = st.sidebar.slider("Overlap Threshold: ", 0.0, 1.0, 0.2, 0.01)

    option = st.sidebar.selectbox("Méthode de détéction", ("Image", "Camera"))

    if option == "Image":
        uploaded_file = st.sidebar.file_uploader("choisir une image", type=['png', 'jpg', 'jpeg'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)

            rotate_direction = st.sidebar.radio("Rotation", ["Original", "90 degrés", "-90 degrés"])
            if rotate_direction == "90 degrés":
                image = image.rotate(-90)
            elif rotate_direction == "-90 degrés":
                image = image.rotate(90)

            if st.button("Prédire"):
                frame = np.array(image)

                # Effectue les prédictions sur l'image pivotée
                preds = model.predict(source=frame, conf=confidence_threshold, iou=overlap_threshold, save=False)

                # Dessine les prédictions sur l'image
                image_with_preds = draw_preds(image, preds)

                # Affiche l'image avec les prédictions
                st.image(image_with_preds, caption="Image prédite", use_column_width=True)

                # Afficher les prédictions
                for i, result in enumerate(preds):
                    st.write("Résultat:")
                    for j, box in enumerate(result.boxes.data):
                        st.write(f"Position (x, y) : ({box[0]}, {box[1]})")
                        st.write(f"Largeur : {box[2]}")
                        st.write(f"Hauteur : {box[3]}")
                        st.write(f"Confiance : {box[4]}")
                        st.write(f"Classe : {result.names[int(box[5])]}")

    elif option == "Camera":
        webrtc_streamer(key="example", video_transformer_factory=lambda: VideoTransformer(confidence_threshold, overlap_threshold))

app()

