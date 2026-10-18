import cv2
from pathlib import Path

try:
    from insightface.app import FaceAnalysis
except ImportError:
    FaceAnalysis = None


# ============================================================
# 1. CONFIGURATION
# ============================================================

DOSSIER = Path(__file__).resolve().parent
DOSSIER_RESULTATS = DOSSIER / "resultats"

DOSSIER_RESULTATS.mkdir(parents=True, exist_ok=True)

modele = None


# ============================================================
# 2. CHARGEMENT DU MODELE
# ============================================================

def charger_modele():
    global modele

    if modele is not None:
        return True

    if FaceAnalysis is None:
        print("Erreur : InsightFace n'est pas installe.")
        return False

    try:
        print("Chargement du modele de detection...")

        nouveau_modele = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )

        nouveau_modele.prepare(
            ctx_id=0,
            det_size=(640, 640)
        )

        modele = nouveau_modele

        print("Modele charge avec succes.")
        return True

    except Exception as erreur:
        print("Impossible de charger le modele.")
        print("Detail :", erreur)
        return False


# ============================================================
# 3. DETECTION DES VISAGES ET IMAGES
# ============================================================

def detecter_visages(image):
    if image is None or image.size == 0:
        return []

    if len(image.shape) != 3 or image.shape[2] != 3:
        return []

    if not charger_modele():
        return []

    try:
        visages_detectes = modele.get(image)
        resultats = []

        for visage in visages_detectes:
            x1, y1, x2, y2 = visage.bbox.astype(int)

            resultats.append({
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2)
            })

        return resultats

    except Exception as erreur:
        print("Erreur pendant la detection :", erreur)
        return []


# ============================================================
# 4. DESSIN DES RECTANGLES
# ============================================================

def dessiner_rectangles(image, visages):
    image_resultat = image.copy()

    for numero, visage in enumerate(visages, start=1):
        x1 = visage["x1"]
        y1 = visage["y1"]
        x2 = visage["x2"]
        y2 = visage["y2"]

        cv2.rectangle(
            image_resultat,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            image_resultat,
            f"Visage {numero}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    if not visages:
        cv2.putText(
            image_resultat,
            "Aucun visage detecte",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    return image_resultat


# ============================================================
# 5. ANALYSE D'UNE PHOTO
# ============================================================

def analyser_image(chemin_image, afficher=True):
    chemin_image = Path(chemin_image)

    image = cv2.imread(str(chemin_image))

    if image is None:
        print("Erreur : image introuvable ou illisible.")
        return []

    visages = detecter_visages(image)

    print("Image :", chemin_image.name)
    print("Nombre de visages detectes :", len(visages))

    for numero, visage in enumerate(visages, start=1):
        print(
            f"Visage {numero} : "
            f"x1={visage['x1']}, "
            f"y1={visage['y1']}, "
            f"x2={visage['x2']}, "
            f"y2={visage['y2']}"
        )

    image_resultat = dessiner_rectangles(image, visages)

    destination = (
        DOSSIER_RESULTATS
        / f"{chemin_image.stem}_detection.jpg"
    )

    if cv2.imwrite(str(destination), image_resultat):
        print("Resultat enregistre :", destination)
    else:
        print("Erreur : impossible d'enregistrer le resultat.")

    if afficher:
        try:
            cv2.imshow("Detection des visages", image_resultat)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            print("Impossible d'afficher la fenetre.")

    return visages


# ============================================================
# 6. DETECTION EN DIRECT AVEC LA WEBCAM
# ============================================================

def utiliser_webcam():
    if not charger_modele():
        return

    print("Ouverture de la webcam...")

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Erreur : impossible d'ouvrir la webcam.")
        camera.release()
        return

    print("Webcam active.")
    print("Place ton visage devant la camera.")
    print("Appuie sur Q pour quitter.")
    print("Appuie sur S pour sauvegarder une capture.")

    try:
        while True:
            succes, image = camera.read()

            if not succes or image is None:
                print("Impossible de lire l'image de la webcam.")
                break

            visages = detecter_visages(image)

            image_resultat = dessiner_rectangles(image, visages)

            cv2.imshow(
                "Detection faciale - Webcam",
                image_resultat
            )

            touche = cv2.waitKey(1) & 0xFF

            if touche == ord("q"):
                break

            if touche == ord("s"):
                destination = (
                    DOSSIER_RESULTATS
                    / "capture_webcam.jpg"
                )

                if cv2.imwrite(str(destination), image_resultat):
                    print("Capture enregistree :", destination)
                else:
                    print("Impossible d'enregistrer la capture.")

    except cv2.error as erreur:
        print("Erreur webcam :", erreur)

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Webcam fermee.")


# ============================================================
# 7. MENU PRINCIPAL
# ============================================================


def main():
    while True:
        print("\n================================")
        print("    DETECTION DES VISAGES")
        print("================================")
        print("1. Analyser une photo")
        print("2. Detecter les visages avec la webcam")
        print("0. Quitter")

        choix = input("Ton choix : ").strip()

        if choix == "1":
            chemin = input("Chemin de la photo : ").strip().strip('"')

            if chemin:
                analyser_image(chemin)

        elif choix == "2":
            utiliser_webcam()

        elif choix == "0":
            print("Fermeture du programme.")
            break

        else:
            print("Choix invalide.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgramme interrompu.")
    except Exception as erreur:
        print("Une erreur inattendue est survenue :", erreur)

