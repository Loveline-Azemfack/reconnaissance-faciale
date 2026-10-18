# 🎓 Système intelligent de reconnaissance faciale des étudiants

## 1. Présentation du projet

Ce projet consiste à développer une application de reconnaissance faciale permettant de détecter les visages des étudiants d'une classe et de rechercher leur identité à partir d'une base de données d'images.

L'application utilise Python, OpenCV et InsightFace pour analyser les images provenant d'une webcam ou chargées depuis l'ordinateur.

Ce travail est réalisé dans un cadre pédagogique afin de mettre en pratique les notions de vision par ordinateur, de traitement d'images, de reconnaissance faciale et de développement collaboratif avec Git et GitHub.

## 2. Objectifs du projet

- Détecter les visages présents dans une image ou dans le flux vidéo d'une webcam.
- Extraire les caractéristiques faciales nécessaires à la comparaison des visages.
- Comparer un visage détecté aux signatures faciales enregistrées dans le dataset.
- Afficher le nom de l'étudiant reconnu ainsi qu'un score de similarité.
- Afficher « Inconnu » lorsqu'aucune identité ne satisfait le seuil de reconnaissance.
- Permettre le traitement d'images importées depuis l'ordinateur.
- Organiser le code en modules indépendants et faciliter le travail en équipe.

## 3. Fonctionnalités

### Détection des visages
Le système utilise InsightFace pour localiser les visages dans les images.

### Reconnaissance faciale
Les caractéristiques faciales extraites sont comparées aux signatures enregistrées pour chaque étudiant.

### Affichage des résultats
L'application est conçue pour afficher :
- Un rectangle autour de chaque visage détecté.
- Le nom de l'étudiant identifié.
- Un score de similarité.
- La mention « Inconnu » lorsque la correspondance est insuffisante.

### Utilisation de la webcam
La webcam permet de capturer des images en direct et de les transmettre au module de traitement.

### Importation d'images
L'utilisateur peut sélectionner une image depuis son ordinateur afin de lancer la détection et la reconnaissance.

### Interface graphique
L'interface permet de lancer la webcam, de charger une image, de consulter les résultats et de fermer l'application.

## 4. Technologies utilisées

| Technologie | Rôle |
|---|---|
| Python | Langage de programmation |
| OpenCV | Capture vidéo et traitement d'images |
| InsightFace | Détection des visages et extraction des caractéristiques faciales |
| NumPy | Calculs numériques et comparaison des signatures |
| Tkinter | Création de l'interface graphique |
| Pillow | Conversion et affichage des images |
| pytest | Exécution des tests automatisés |
| Git et GitHub | Gestion des versions et collaboration |

## 5. Architecture du projet

```text
reconnaissance-faciale/
│
├── dataset/
│   ├── Durel/
│   ├── Emma/
│   ├── Gabriel/
│   ├── Herve/
│   └── Loveline/
│
├── src/
│   ├── __init__.py
│   ├── camera.py
│   ├── data_preparation.py
│   ├── face_detection.py
│   ├── face_recognition.py
│   ├── interface.py
│   └── main.py
│
├── tests/
│   └── test_camera.py
│
├── notebook/
│   └── entrainement.ipynb
│
├── .gitignore
├── README.md
└── requirements.txt
```

*Remarque : cette arborescence présente l'organisation prévue du projet. Certains fichiers peuvent être ajoutés ou adaptés selon l'état final du dépôt.*

### Rôle des principaux fichiers

- `src/main.py` : point d'entrée de l'application et coordination entre l'interface et le module de caméra.
- `src/interface.py` : création de la fenêtre graphique et affichage des images et des résultats.
- `src/camera.py` : ouverture de la webcam, capture des images, préparation des signatures et reconnaissance faciale.
- `src/face_detection.py` : chargement du modèle et détection des visages.
- `src/face_recognition.py` : fonctions liées à la comparaison et à l'identification des visages.
- `src/data_preparation.py` : fonctions destinées à la préparation et à l'organisation des données.
- `tests/` : tests automatisés du projet.
- `notebook/entrainement.ipynb` : notebook pédagogique associé au projet.

## 6. Installation

### Prérequis

Avant de commencer, il faut disposer de :

- Python installé sur l'ordinateur.
- Git installé.
- Une webcam fonctionnelle pour tester la vidéo en direct.
- Une connexion Internet pour installer les dépendances et, si nécessaire, télécharger les modèles utilisés par InsightFace.

### Étape 1 : Cloner le dépôt

```bash
git clone https://github.com/Loveline-Azemfack/reconnaissance-faciale.git
cd reconnaissance-faciale
```

### Étape 2 : Créer un environnement virtuel

Sous Windows :

```powershell
py -m venv .venv
```

Activer l'environnement :

```powershell
.\.venv\Scripts\Activate.ps1
```

### Étape 3 : Installer les dépendances

Si le fichier `requirements.txt` est présent et à jour :

```powershell
py -m pip install -r requirements.txt
```

Si certaines dépendances sont manquantes, vérifier les imports et la configuration du projet avant de les installer individuellement.

## 7. Exécution de l'application

Depuis la racine du projet, lancer :

```powershell
py -m src.main
```

La fenêtre principale de l'application devrait s'ouvrir.

L'utilisateur pourra ensuite :
1. Ouvrir la webcam.
2. Observer les images capturées.
3. Consulter les résultats de reconnaissance.
4. Charger une image pour effectuer un test.
5. Fermer la webcam ou quitter l'application.

Pour lancer directement le module de caméra, si nécessaire :

```powershell
py -m src.camera
```

## 8. Organisation du dataset

Le dataset regroupe les images des étudiants, avec un dossier par personne.

Exemple :

```text
dataset/
├── Emma/
│   ├── photo_01.jpg
│   ├── photo_02.jpg
│   └── ...
├── Durel/
├── Gabriel/
├── Herve/
└── Loveline/
```

Les images doivent présenter autant que possible des variations de :
- Orientation du visage.
- Expressions faciales.
- Distance par rapport à la caméra.
- Conditions d'éclairage.

Le projet prévoit environ 20 à 40 images par étudiant, conformément aux consignes pédagogiques.

Les images doivent être collectées avec le consentement des personnes concernées. Il est recommandé de limiter l'accès au dataset et de ne pas diffuser les photos sans autorisation.

## 9. Principe de fonctionnement

Le traitement suit les étapes suivantes :

1. **Acquisition** : capture d'une image par la webcam ou importation d'une photographie.
2. **Détection** : localisation des visages dans l'image.
3. **Extraction** : calcul d'une signature numérique pour chaque visage.
4. **Comparaison** : comparaison de cette signature aux signatures enregistrées dans le dataset.
5. **Identification** : sélection de la meilleure correspondance.
6. **Décision** : attribution d'un nom si le seuil défini est atteint, sinon affichage de « Inconnu ».
7. **Visualisation** : affichage des rectangles, des noms et des scores disponibles.

## 10. Tests

Pour lancer les tests automatisés :

```powershell
py -m pytest -v
```

Cette commande permet de vérifier les fonctionnalités couvertes par les tests.

Des essais manuels doivent également être réalisés pour vérifier :
- L'ouverture et la fermeture de la webcam.
- La détection des visages en direct.
- La reconnaissance d'étudiants présents dans le dataset.
- Le comportement avec une personne absente du dataset.
- Le chargement et le traitement d'une image.
- L'affichage des rectangles et des noms dans l'interface.
- La réactivité de l'application pendant le traitement vidéo.

Les tests automatisés ne remplacent pas les essais fonctionnels de l'application complète.

## 11. Collaboration avec Git et GitHub

Le projet est développé de manière collaborative à l'aide de branches Git.

Pour consulter les branches locales :

```bash
git branch
```

Pour récupérer les dernières modifications du dépôt :

```bash
git fetch origin
```

Pour vérifier les modifications locales :

```bash
git status
```

Avant de fusionner les contributions, il est recommandé d'exécuter les tests et de résoudre les éventuels conflits avec les membres concernés.

## 12. Limites et améliorations possibles

Plusieurs améliorations peuvent être envisagées :

- Optimiser la fluidité de la vidéo en direct.
- Améliorer la réactivité de l'interface pendant la reconnaissance.
- Tester et ajuster le seuil de reconnaissance sur des images indépendantes.
- Renforcer la robustesse face aux changements d'éclairage et d'orientation.
- Élargir le dataset avec des images variées et autorisées.
- Améliorer les tests d'intégration entre la caméra, la reconnaissance et l'interface.

Le score affiché est un score de similarité, et non une probabilité garantie d'identité. Les résultats doivent être évalués à partir de tests représentatifs.

## 13. Confidentialité et utilisation responsable

Ce projet est destiné à l'apprentissage de la vision par ordinateur.

Les données faciales sont sensibles. Leur collecte, leur utilisation et leur partage doivent respecter le consentement des participants et les règles applicables en matière de protection des données.

Le système ne doit pas être utilisé comme unique moyen de vérifier l'identité d'une personne dans une situation importante.

## 14. Conclusion

Ce projet permet de mettre en pratique les techniques de détection et de reconnaissance faciales, le traitement d'images en temps réel et le développement collaboratif en Python.

Il constitue une base pédagogique pouvant être améliorée progressivement grâce aux tests, à l'optimisation des performances et à l'intégration des différents modules.

---

**Projet :** Reconnaissance faciale intelligente des étudiants  
**Dépôt GitHub :** https://github.com/Loveline-Azemfack/reconnaissance-faciale  
**Branche de travail caméra :** `feature/camera`