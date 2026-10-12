
import cv2


def display_result(frame, faces):
    """Affiche les rectangles, les noms et les scores sur l'image."""

    for face in faces:
        x, y, largeur, hauteur = face["box"]
        nom = face.get("name", "Inconnu")
        confiance = face.get("confidence")

        # Dessiner le rectangle autour du visage
        cv2.rectangle(
            frame,
            (x, y),
            (x + largeur, y + hauteur),
            (0, 255, 0),
            2
        )

        # Préparer le texte à afficher
        if confiance is not None:
            texte = f"{nom} - {confiance:.1f}%"
        else:
            texte = nom

        # Afficher le nom et le score au-dessus du rectangle
        cv2.putText(
            frame,
            texte,
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return frame
def open_camera():
    """Ouvre la webcam et vérifie qu'elle est disponible."""
    webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not webcam.isOpened():
        raise RuntimeError("Impossible d'ouvrir la webcam.")

    return webcam


def capture_frame(webcam):
    """Capture une image depuis la webcam."""
    ret, frame = webcam.read()

    if not ret:
        return None

    return frame


def run_camera():
    webcam = None

    try:
        webcam = open_camera()
        print("Webcam active ! Appuie sur q pour quitter.")

        while True:
            frame = capture_frame(webcam)

            if frame is None:
                print("Impossible de récupérer une image.")
                break

            print("Dimensions :", frame.shape)

            # Aucun résultat réel n'est encore connecté
            faces = []

            frame = display_result(frame, faces) 

            frame = display_result(frame, faces)

            cv2.imshow("Projet de reconnaissance faciale", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except RuntimeError as erreur:
        print(erreur)

    finally:
        if webcam is not None:
            webcam.release()

        cv2.destroyAllWindows()
        print("Webcam fermée.")


if __name__ == "__main__":
    run_camera()
