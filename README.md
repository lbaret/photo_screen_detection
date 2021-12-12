# Projet de détection de capture d’écran

![photo](/img/example_photo.jpg) 

**vs.** 

![screenshot](/img/example_screenshot.png)

## Environnement Python

### Installation

#### Pyenv
```shell
pyenv install 3.9.6
pyenv local 3.9.6
```

#### Poetry
```shell
poetry install
```

### Usage
```shell
poetry shell
poetry run
```

## Partie Machine Learning
Ce projet a pour premier but de classifier des images en deux catégories : 
1. Screenshot
2. Photo

## Motivations
Ensuite, il serait question de construire un bot Discord pour détecter
les photographies d’un écran afin de rendre le contenu du serveur plus qualitatif.

Pour se faire, j’ai collecté 1000 images composées de 500 captures d’écran et 500 photos provenant
de divers groupes facebooks publics orientés programmation.

## Twitch - Diffusion en direct
Ce projet sera diffusé via la plateforme *twitch*, sur ma chaîne dédiée au *Machine Learning*.  
Vous avez la possibilité de suivre l’évolution du projet en direct en cliquant sur [ce lien](https://www.twitch.tv/lobiten).

*P.S. : n’hésitez surtout pas à rejoindre le direct afin de poser toutes vos questions et venir discuter
de l’Intelligence Artificielle en général*

## Organisation du code
```
|-- Projet - racine
|  |-- photo_screen_detection : module 
|  |  |-- notebooks
|  |  |  |-- eda.py : rapide affichage des données que nous possédons.
|  |  |  |-- baseline.py : on utilise des modèles ayant fais leur preuve dans des projets similaires.
|  |  |  |-- garbage.py : fichier pour tester des morceaux de codes (foure tout).
|  |  |-- config
|  |  |  |-- paths.py : contient les différents chemins d'accès
|  |-- data : répertoire contenant les données.
|  |  |-- photo
|  |  |-- screenshot
|  |-- img : répertoire contenant les images affichés dans le README.md
|-- pyproject.toml : fichier de configuration Poetry
|-- poetry.lock : fichier généré pour la reproduction de l'environnement virtuel Poetry
```

## Avancées
Pour l’instant, j’ai simplement implémenté et ajusté un modèle [ResNet-18](https://arxiv.org/abs/1512.03385) pré-entrainé.

## Idées :bulb:
La grande ligne directrice est :
1. Utiliser des réseaux pré-entrainés.
2. Construire un réseau de neurones basé sur des couches convolutionnelles.
3. Implémenter un algorithme de recherche d’hyperparamètres.
4. Lier le modèle avec un bot discord.
