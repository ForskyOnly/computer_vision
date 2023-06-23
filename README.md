# Computer Vision Détection d'objet 


## Context du projet

Vous êtes désormais des ingénieurs en Machine Learning au sein de SightAI, une entreprise leader dans le domaine de la vision par ordinateur, réputée pour son innovation dans l'IA. Votre dernière mission a été de concevoir et d'entraîner des réseaux de neurones à partir de zéro ainsi que de réaliser du Transfer Learning sur des modèles pré-entraînés. Suite à la réussite de cette mission, vous voilà prêts pour un défi encore plus grand.

Votre nouvelle mission, si vous l'acceptez, sera de concevoir une application qui intègre YOLO-v8, le dernier modèle d'IA pour la vision par ordinateur, le plus performant et sophistiqué de 2023. Cette application devra résoudre un problème de votre choix, dans le domaine de la vision par ordinateur, que vous aurez défini vous-mêmes. Vous devrez développer une application "quick and dirty", votre objectif est d'avoir une application fonctionnelle dans un délai court. En effet, c'est encore votre période d'essaie et votre manager souhaite tester votre capacité à implémenter un modèle dans une application simple.

Pour cela, voici les étapes que vous devrez suivre :

Choisir une problématique : Identifiez une problématique ou un besoin spécifique auquel votre application répondra. Vous vous baserez sur les datasets disponibles sur Roboflow pour trouver des pistes faisables.

​
Mise en œuvre de YOLO-v8 : Utilisez YOLO-v8 comme base pour votre modèle. Commencez par utiliser les poids pré-entraînés de COCO pour initialiser votre modèle.


Récupération et préparation du dataset : Pour entraîner votre modèle à votre problème spécifique, vous utiliserez un ensemble de données que vous récupérerez sur Roboflow. Assurez-vous de préparer et de nettoyer correctement vos données pour l'entraînement.

Transfer Learning : Effectuez du Transfer Learning en ré-entraînant votre modèle YOLO-v8 sur votre ensemble de données.

Création d'une application avec Streamlit : Utilisez Streamlit pour développer une application qui met en œuvre votre modèle entraîné. Votre application doit intégrer l'une des fonctionnalités suivantes :

- Permettre d'upload une image ou une vidéo et d'avoir les prédictions. (bounding boxes et classes)
- Avoir des prédictions en live en utilisant la webcam de l'utilisateur.

​
L'application doit être simple, mais efficace (une approche "quick and dirty").


Test et déploiement de l'application : Une fois votre application développée, testez-la pour vous assurer qu'elle fonctionne correctement et déployez-la sur Azure à l'aide de docker.

# La problématique: 

Pour la problématique nous avons opté pour les lettres du langage des singes:
- le modéle entrainer prends en entré un signe sous forme d'image ou en direct via la caméra de l'utilisateur et renvoi la lettre qui corresponds au signe 

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


