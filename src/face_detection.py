import cv2
from pathlib import Path

try:
    from insightface.app import FaceAnalysis
except ImportError:
    FaceAnalysis = None

modele = None


def charger_modele():    

    global modele

    if modele is not None:
        return True

    if FaceAnalysis is None:
        print("InsightFace n'est pas installe.")
        return False

    try:
        modele = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )
        modele.prepare(ctx_id=-1, det_size=(640, 640))
        return True
    
    except Exception as erreur:
        modele = None
        print("Erreur de chargement :", erreur)
        return False


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
        visages = modele.get(image)
        resultats = []

        for visage in visages:
            x1, y1, x2, y2 = visage.bbox.astype(int)

            resultats.append({
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2)
            })

        return resultats
    
    except Exception as erreur:
        print("Erreur de detection :", erreur)
        return []


def dessiner_rectangles(image, visages):
    
    if image is None:
        return None

    resultat = image.copy()

    for visage in visages:
        cv2.rectangle(
            resultat,
            (visage["x1"], visage["y1"]),
            (visage["x2"], visage["y2"]),
            (0, 255, 0),
            2
        )

    return resultat
