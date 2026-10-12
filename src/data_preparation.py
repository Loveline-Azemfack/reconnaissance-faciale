# Importe Path pour manipuler les chemins des dossiers et fichiers.
from pathlib import Path 
# Importe Image de Pillow pour vérifier les fichiers image.
from PIL import Image #pillow est une bibliotheque de python qui me permet de manipuler de verifier les images

# Définit le chemin du dossier dataset à partir de l'emplacement de ce fichier.
DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"
#cette ligne dit avec __file__ represente le fichier actuel , avec .resolve() recupere son chemin absolu, .parent remonte a son dossier direct dans lequel il est inclu et c'est src, 
#on va donc encore ajouter .parent pourque ca monte ancore dans le dossier parent et ici c'est reconnaissance_faciale, et lorsqu'on ajoute /"dataset" c'est pour dire qu'a parir de la accede au dossier dataset

# Définit la fonction de vérification du dataset.
def verify_dataset():
    # Vérifie si le dossier dataset existe.
    if not DATASET_DIR.exists():
        # Affiche le chemin du dossier introuvable.
        print(f"Erreur : dataset introuvable : {DATASET_DIR}")
        # Arrête la fonction si le dossier n'existe pas.
        return

    # Initialise le nombre total d'images examinées.
    total_images = 0
    # Initialise le nombre d'images valides.
    valid_images = 0
    # Initialise le nombre d'images invalides.
    invalid_images = 0

    # Parcourt les éléments du dataset par ordre alphabétique.
    #le .iterdir()  c'est pour recuperer les elements du dossier, sorted permet de classer par ordre alphabetic
    for student_dir in sorted(DATASET_DIR.iterdir()):
        # Ignore les éléments qui ne sont pas des dossiers.
        if not student_dir.is_dir():  #si l'element rechercher est un dossier, alors on continue donc is_dir() devient true
            continue

        # Recherche les fichiers JPG, JPEG et PNG dans le dossier étudiant.
        image_files = [
            # Récupère le chemin de chaque image correspondante.
            path
            # Parcourt les éléments du dossier étudiant.
            for path in student_dir.iterdir() #donc suite au test avec is_dir, on entre donc dans chaque dossier pour effectuer l'analyse des fichiers
            # Vérifie que l'élément est un fichier image accepté.
            if path.is_file()
            and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]

        # Vérifie si aucune image n'a été trouvée.
        if not image_files:
            # Signale que le dossier ne contient aucune image reconnue.
            print(f"{student_dir.name} : aucune image trouvée")
            # Passe au dossier suivant.
            continue

        # Affiche le nom de l'étudiant en cours de vérification.
        print(f"\nÉtudiant : {student_dir.name}")

        # Parcourt les images trouvées dans le dossier.
        for image_path in image_files:
            # Compte l'image examinée.
            total_images += 1

            # Tente d'ouvrir et de vérifier l'image.
            try:
                # Ouvre l'image et ferme automatiquement le fichier après utilisation.
                with Image.open(image_path) as image:
                    # Vérifie l'intégrité du fichier image.
                    image.verify()

                # Compte l'image comme valide si la vérification réussit.
                valid_images += 1
                # Affiche le nom de l'image valide.
                print(f"  OK : {image_path.name}")

            # Intercepte les erreurs de lecture ou de valeur.
            except (OSError, ValueError) as error:
                # Compte l'image comme invalide.
                invalid_images += 1
                # Affiche le nom de l'image et la raison de l'erreur.
                print(f"  ERREUR : {image_path.name} ({error})")

    # Affiche le titre du bilan final.
    print("\n========== BILAN DU DATASET ==========")
    # Affiche le nombre total d'images examinées.
    print(f"Images examinées : {total_images}")
    # Affiche le nombre d'images valides.
    print(f"Images valides   : {valid_images}")
    # Affiche le nombre d'images invalides.
    print(f"Images invalides : {invalid_images}")


# Vérifie si le fichier est exécuté directement.
if __name__ == "__main__":
    # Lance la vérification du dataset.
    verify_dataset()