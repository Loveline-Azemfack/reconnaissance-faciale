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
        print("Erreur : InsightFace n'est pas installé.")
        return False

    try:
        modele_temporaire = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )

        modele_temporaire.prepare(
            ctx_id=0,
            det_size=(640, 640)
        )

        modele = modele_temporaire

        return True

    except Exception as erreur:
        print("Impossible de charger le modèle.")
        print("Vérifie les dépendances et la connexion Internet.")
        print("Détail :", erreur)
        return False


# ============================================================
# 3. DETECTION DES VISAGES
# ============================================================

def detecter_visages(image):
    

    if image is None:
        print("Image invalide.")
        return []

    if not hasattr(image, "shape") or image.size == 0:
        print("Image vide ou invalide.")
        return []

    if len(image.shape) != 3 or image.shape[2] != 3:
        print("Format d'image non pris en charge.")
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
        print("La détection des visages a échoué.")
        print("Détail :", erreur)
        return []


# ============================================================
# 4. DESSIN DES RECTANGLES
# ============================================================

def dessiner_rectangles(image, visages):
    

    if image is None:
        return None

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
            0.6,
            (0, 255, 0),
            2
        )

    return image_resultat


# ============================================================
# 5. ANALYSE D'UNE IMAGE
# ============================================================

def analyser_image(chemin_image, afficher=True):
    

    chemin_image = Path(chemin_image)

    try:
        image = cv2.imread(str(chemin_image))
    except (cv2.error, OSError):
        print("Impossible de lire l'image.")
        return []

    if image is None:
        print("Image introuvable ou fichier illisible.")
        return []

    visages = detecter_visages(image)

    print("Image :", chemin_image.name)
    print("Nombre de visages détectés :", len(visages))

    for numero, visage in enumerate(visages, start=1):
        print(
            f"Visage {numero} : "
            f"x1={visage['x1']}, "
            f"y1={visage['y1']}, "
            f"x2={visage['x2']}, "
            f"y2={visage['y2']}"
        )

    if not visages:
        print("Aucun visage détecté ou détection indisponible.")

    image_resultat = dessiner_rectangles(image, visages)

    if image_resultat is None:
        print("Impossible de préparer l'image résultat.")
        return visages

    destination = (
        DOSSIER_RESULTATS
        / f"{chemin_image.stem}_detection.jpg"
    )

    try:
        sauvegarde_reussie = cv2.imwrite(
            str(destination),
            image_resultat
        )
    except (cv2.error, OSError):
        sauvegarde_reussie = False

    if sauvegarde_reussie:
        print("Résultat enregistré :", destination)
    else:
        print("Impossible d'enregistrer l'image résultat.")

    if afficher:
        try:
            cv2.imshow("Detection des visages", image_resultat)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            print("Affichage indisponible sur cet environnement.")
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    return visages


# ============================================================
# 6. PROGRAMME PRINCIPAL
# ============================================================

def main():
    

    chemin = input(
        "Chemin de l'image à analyser : "
    ).strip().strip('"')

    if not chemin:
        print("Aucune image sélectionnée.")
        return

    try:
        analyser_image(chemin)
    except Exception as erreur:
        print("Une erreur inattendue est survenue.")
        print("Détail :", erreur)


if __name__ == "__main__":
    main()
