# Computer Vision Détection d'objet 


## Context du projet

Le projet "Vision par ordinateur pour la Langue des Signes" consiste à développer un système capable de reconnaître et d'interpréter les gestes de la Langue des Signes à partir d'images ou de vidéos. En utilisant des algorithmes de détection de mouvement et des modèles d'apprentissage profond, notre solution permettra une reconnaissance en temps réel des signes, facilitant ainsi la communication entre les personnes sourdes et entendantes. Notre objectif est de favoriser l'inclusion et la compréhension mutuelle, en contribuant à un monde plus accessible pour tous.


# La problématique: 

Pour la problématique nous avons opté pour les lettres du langage des singes:
- le modéle entrainer prends en entré un signe sous forme d'image ou en direct via la caméra de l'utilisateur et renvoi la lettre qui corresponds au signe 

## Etapes projet  : 
    
- Choix d'une problématique : Identifiez un besoin spécifique dans le domaine de la vision par ordinateur que votre application résoudra.
- Mise en œuvre de YOLO-v8 : Utilisez le modèle YOLO-v8 comme base pour votre application, en utilisant les poids pré-entraînés de COCO pour l'initialisation.
- Récupération et préparation du dataset : Obtenez un ensemble de données approprié à votre problème depuis Roboflow, puis préparez et nettoyez les données pour l'entraînement.
- Transfer Learning : Ré-entraînez le modèle YOLO-v8 en utilisant votre propre ensemble de données spécifique à votre problème.
- Création d'une application avec Streamlit : Développez une application utilisant Streamlit, qui met en œuvre votre modèle entraîné et permet soit de faire des prédictions à partir d'images/vidéos uploadées, soit d'avoir des prédictions en direct à partir de la webcam de l'utilisateur.
- Test et déploiement de l'application : Testez soigneusement l'application pour vous assurer de son bon fonctionnement, puis déployez-la sur Azure en utilisant Docker.

## Fichiers présent dans le depot: 

- best.pt : Ce fichier contient le modéle entrainer pour détecter les signes 
- sing_language.ipynb : Ce fichier contient l'import du dataset, le modéle entrainer pour la problématique et quelques visualisation concernant les score obtenu par le modéle 
- requirements.txt : Ce fichier contient les bibliothéque nécessaire pour installer et utiliser l'application
- main.py : Ce fichier contient le programme principal de l'application streamlit 

## Installation

1. Clonez [ce dépôt.](https://github.com/ForskyOnly/computer_vision)
2. Installez les dépendances avec `pip install -r requirements.txt`.


## Bibliothèque utilisées

- streamlit
- opencv-python
- ultralytics
- streamlit-webrtc



## Utilisation

- installer les dépendances nécessaires
- lancer l'application avec la commande `streamlit run main.py`
- télecharger une image contenant un signe ou acceder a la caméra pour effectuer les détection

**Vous pouvez désormer détecter les signes et renvoyer la lettre qui leurs sont propres**


## License

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

## Credit

- le dataset à été telecharger depuis le site de Roboflow :

`@misc{ sign-language-sokdr_dataset,
    title = { sign language Dataset },
    type = { Open Source Dataset },
    author = { Roboflow 100 },
    howpublished = { \url{ https://universe.roboflow.com/roboflow-100/sign-language-sokdr } },
    url = { https://universe.roboflow.com/roboflow-100/sign-language-sokdr },
    journal = { Roboflow Universe },
    publisher = { Roboflow },
    year = { 2023 },
    month = { may },
    note = { visited on 2023-06-23 },
}`
- Le modéle (YOLOv8) a été choisi depuis le depot github de [ultralytics](https://github.com/ultralytics/ultralytics):

## Contributeurs: 

- [Rubic](https://github.com/ForskyOnly)
- [Noura](https://github.com/Noura-ou)

## Pistes d'amelioration :

Dans le but d'améliorer la performance du modèle, une suggestion serait d'enrichir son entraînement en ajoutant davantage de données. Même si l'application fonctionne correctement, il est parfois évident que le modèle a du mal à détecter les signes. En intégrant plus de données d'entraînement, le modèle aura la possibilité d'apprendre à reconnaître une plus grande variété de signes et devrait donc être en mesure de les détecter plus précisément. Cela pourrait se traduire par une meilleure performance globale de l'application et une meilleure capacité à détecter les signes dans différents contextes.


