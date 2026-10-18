
import cv2
import numpy as np
import threading
import time
from pathlib import Path

try:
    from . import face_detection
except ImportError:
    import face_detection


DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"

# Seuil de similarité à ajuster après les tests.
SEUIL_RECONNAISSANCE = 0.45

webcam = None
signatures_connues = None


def display_result(frame, faces):
    """Dessine les rectangles, les noms et les scores."""
    for face in faces:
        x, y, largeur, hauteur = face["box"]
        nom = face.get("name", "Inconnu")
        confiance = face.get("confidence")

        cv2.rectangle(
            frame, (x, y), (x + largeur, y + hauteur),
            (0, 255, 0), 2
        )

        if confiance is not None:
            texte = f"{nom} - {confiance * 100:.1f}%"
        else:
            texte = nom

        cv2.putText(
            frame, texte, (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

    return frame


def open_camera():
    """Ouvre la webcam et renvoie True si elle fonctionne."""
    global webcam

    if webcam is not None and webcam.isOpened():
        return True

    webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not webcam.isOpened():
        webcam.release()
        webcam = None
        return False

    return True


def capture_frame():
    """Capture une image depuis la webcam."""
    if webcam is None or not webcam.isOpened():
        return None

    ret, frame = webcam.read()
    return frame if ret else None


def close_camera():
    """Libère la webcam."""
    global webcam

    if webcam is not None:
        webcam.release()
        webcam = None


def preparer_signatures():
    """Calcule les signatures faciales des photos du dataset."""
    global signatures_connues

    if signatures_connues is not None:
        return signatures_connues

    if not face_detection.charger_modele():
        return {}

    signatures_connues = {}

    if not DATASET_DIR.is_dir():
        print(f"Dataset introuvable : {DATASET_DIR}")
        return signatures_connues

    extensions = {".jpg", ".jpeg", ".png"}

    for dossier in sorted(DATASET_DIR.iterdir()):
        if not dossier.is_dir():
            continue

        signatures = []

        for chemin in dossier.iterdir():
            if not chemin.is_file() or chemin.suffix.lower() not in extensions:
                continue

            image = cv2.imread(str(chemin))
            if image is None:
                continue

            visages = face_detection.modele.get(image)

            if not visages:
                continue

            # Utilise le visage le plus grand dans la photo.
            visage = max(
                visages,
                key=lambda v: (
                    v.bbox[2] - v.bbox[0]
                ) * (
                    v.bbox[3] - v.bbox[1]
                )
            )

            signatures.append(
                visage.embedding / np.linalg.norm(visage.embedding)
            )

        if signatures:
            signatures_connues[dossier.name] = signatures
            print(f"{dossier.name} : {len(signatures)} signatures préparées")
        else:
            print(f"{dossier.name} : aucune signature exploitable")

    return signatures_connues


def process_frame(frame):
    """Détecte les visages et recherche leur identité."""
    if frame is None or frame.size == 0:
        return []

    if not face_detection.charger_modele():
        return []

    base = preparer_signatures()
    detections = face_detection.modele.get(frame)
    resultats = []

    for visage in detections:
        x1, y1, x2, y2 = map(int, visage.bbox)
        nom = "Inconnu"
        confiance = 0.0

        signature = visage.embedding
        norme = np.linalg.norm(signature)

        if norme > 0 and base:
            signature = signature / norme
            meilleur_score = -1.0
            meilleur_nom = None

            for nom_etudiant, signatures in base.items():
                scores = [
                    float(np.dot(signature, connue))
                    for connue in signatures
                ]
                score_etudiant = max(scores)

                if score_etudiant > meilleur_score:
                    meilleur_score = score_etudiant
                    meilleur_nom = nom_etudiant

            if meilleur_nom is not None:
                confiance = max(0.0, min(1.0, meilleur_score))

                if meilleur_score >= SEUIL_RECONNAISSANCE:
                    nom = meilleur_nom

        resultats.append({
            "bbox": (x1, y1, x2 - x1, y2 - y1),
            "box": (x1, y1, x2 - x1, y2 - y1),
            "name": nom,
            "confidence": confiance
        })

    return resultats


def draw_results(frame, results):
    """Dessine les résultats sur l'image."""
    return display_result(frame, results)




def run_camera():
    """Affiche la vidéo pendant que la reconnaissance travaille en arrière-plan."""
    if not open_camera():
        print("Impossible d'ouvrir la webcam.")
        return

    verrou = threading.Lock()
    arret = threading.Event()
    etat = {"frame": None, "resultats": []}

    def reconnaitre_en_arriere_plan():
        derniere_image = None

        while not arret.is_set():
            with verrou:
                image = (
                    etat["frame"].copy()
                    if etat["frame"] is not None
                    else None
                )

            if image is not None:
                if derniere_image is None or not np.array_equal(
                    image, derniere_image
                ):
                    try:
                        resultats = process_frame(image)
                        with verrou:
                            etat["resultats"] = resultats
                        derniere_image = image
                    except Exception as erreur:
                        print("Erreur de reconnaissance :", erreur)

            time.sleep(0.05)

    travailleur = threading.Thread(
        target=reconnaitre_en_arriere_plan,
        daemon=True
    )
    travailleur.start()

    print("Webcam active ! Appuie sur q pour quitter.")

    try:
        while True:
            frame = capture_frame()

            if frame is None:
                break

            with verrou:
                etat["frame"] = frame.copy()
                resultats = list(etat["resultats"])

            image_annotee = draw_results(frame.copy(), resultats)
            cv2.imshow("Projet de reconnaissance faciale", image_annotee)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        arret.set()
        travailleur.join(timeout=2)
        close_camera()
        cv2.destroyAllWindows()
        print("Webcam fermée.")




if __name__ == "__main__":
    run_camera()
