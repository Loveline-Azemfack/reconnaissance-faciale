
import cv2


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
            print("Luminosité moyenne :", frame.mean())    

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
