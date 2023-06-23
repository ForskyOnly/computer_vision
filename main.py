from ultralytics import YOLO
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw

def charger_model():
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
        self.model = charger_model()
        self.confidence_threshold = confidence_threshold
        self.overlap_threshold = overlap_threshold

    def transform(self, cadre):
        """
        Transformer un cadre de la vidéo en détectant les objets à l'aide du modèle YOLO et en dessinant les résultats.

        Args:
        cadre (np.ndarray): cadre de la vidéo à analyser.

        Returns:
        img (np.ndarray): Image avec les prédictions dessinées.
        """
        img = cadre.to_ndarray(format="bgr24")

        # Applique le modèle
        preds = self.model.predict(source=img, conf=self.confidence_threshold, iou=self.overlap_threshold, save=False)

        # Dessine les résultats sur l'image
        for i, resultat in enumerate(preds):
            for j, box in enumerate(resultat.boxes.data):
                cv2.rectangle(img,
                              (int(box[0]), int(box[1])),
                              (int(box[0] + box[2]), int(box[1] + box[3])),
                              (255, 0, 0),
                              2)
                cv2.putText(img, 
                            f"Classe : {resultat.names[int(box[5])]}, Confiance : {box[4]}", 
                            (int(box[0]), int(box[1]-10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.9, 
                            (36, 255, 12), 
                            2)

        return img

def dessiner_predictions(image, preds):
    """
    Dessiner les prédictions sur l'image.

    Args:
    image (PIL.Image): Image sur laquelle dessiner les prédictions.
    preds (list): Liste de prédictions à dessiner.

    Returns:
    image (PIL.Image): Image avec les prédictions dessinées.
    """
    dessine = ImageDraw.Draw(image)

    for i, resultat in enumerate(preds):
        for j, box in enumerate(resultat.boxes.data):
            # Convertie les coordonnées des boites
            x1, y1 = int(box[0]), int(box[1])
            x2, y2 = int(box[0] + box[2]), int(box[1] + box[3])
            
            # Dessine le rectangle
            dessine.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
            text = f"Classe : {resultat.names[int(box[5])]}, Confiance : {box[4]}"
            
            # Ajoute le texte
            dessine.text((x1, y1-10), text, fill=(36, 255, 12))

    return image

def app():
    """
    Fonction principale pour l'application Streamlit. Gère l'interaction avec l'utilisateur et affiche les résultats.
    """
    st.title("Alphabet du langage des signes")
    st.sidebar.title("Réglages")

    model = charger_model()

    # Paramètres CONF et IOU
    confidence_threshold = st.sidebar.slider("Confidence Threshold:", 0.0, 1.0, 0.3, 0.01)
    overlap_threshold = st.sidebar.slider("Overlap Threshold: ", 0.0, 1.0, 0.2, 0.01)

    option = st.sidebar.selectbox("Méthode de détéction", ("Image", "Camera"))

    if option == "Image":
        charge_img = st.sidebar.file_uploader("choisir une image", type=['png', 'jpg', 'jpeg'])
        if charge_img is not None:
            image = Image.open(charge_img)

            direction_rota = st.sidebar.radio("Rotation", ["Original", "90 degrés", "-90 degrés"])
            if direction_rota == "90 degrés":
                image = image.rotate(-90)
            elif direction_rota == "-90 degrés":
                image = image.rotate(90)

            if st.button("Prédire"):
                cadre = np.array(image)

                # Effectue les prédictions sur l'image pivotée
                preds = model.predict(source=cadre, conf=confidence_threshold, iou=overlap_threshold, save=False)

                # Dessine les prédictions sur l'image
                image_avec_pred = dessiner_predictions(image, preds)

                # Affiche l'image avec les prédictions
                st.image(image_avec_pred, caption="Image prédite", use_column_width=True)

                # Afficher les prédictions
                for i, resultat in enumerate(preds):
                    st.write("Résultat:")
                    for j, box in enumerate(resultat.boxes.data):
                        st.write(f"Position (x, y) : ({box[0]}, {box[1]})")
                        st.write(f"Largeur : {box[2]}")
                        st.write(f"Hauteur : {box[3]}")
                        st.write(f"Confiance : {box[4]}")
                        st.write(f"Classe : {resultat.names[int(box[5])]}")

    elif option == "Camera":
        webrtc_streamer(key="example", video_transformer_factory=lambda: VideoTransformer(confidence_threshold, overlap_threshold))

app()

