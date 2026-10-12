# Importe Path pour manipuler les chemins des dossiers et des fichiers.
from pathlib import Path

# Importe Image de Pillow pour ouvrir et vérifier les images.
from PIL import Image

# Importe le chemin du dataset et la fonction de vérification du fichier data_preparation.py.
from src.data_preparation import DATASET_DIR, verify_dataset


# Vérifie que le dossier dataset existe.
def test_dataset_exists():
    # Le test réussit si le chemin du dataset existe.
    assert DATASET_DIR.exists()


# Vérifie que le dataset contient au moins un dossier étudiant.
def test_student_directories_exist():
    # Récupère les éléments qui sont des dossiers dans le dataset.
    student_dirs = [
        path for path in DATASET_DIR.iterdir()
        if path.is_dir()
    ]
    # Vérifie qu'au moins un dossier étudiant a été trouvé.
    assert len(student_dirs) > 0


# Vérifie que les fichiers image du dataset peuvent être ouverts et contrôlés.
def test_images_are_valid():
    # Parcourt les éléments présents dans le dossier dataset.
    for student_dir in DATASET_DIR.iterdir():
        # Ignore les éléments qui ne sont pas des dossiers.
        if not student_dir.is_dir():
            continue

        # Parcourt les éléments présents dans chaque dossier étudiant.
        for image_path in student_dir.iterdir():
            # Vérifie que l'élément est un fichier avec une extension d'image acceptée.
            if image_path.is_file() and image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                # Ouvre l'image et ferme automatiquement le fichier après utilisation.
                with Image.open(image_path) as image:
                    # Vérifie l'intégrité du fichier image.
                    image.verify()