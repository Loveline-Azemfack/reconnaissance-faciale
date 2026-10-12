
import cv2

for index in range(5):
    print(f"\nTest de la caméra {index}")

    camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)

    if not camera.isOpened():
        print("Impossible d'ouvrir cette caméra.")
        camera.release()
        continue

    ret, frame = camera.read()

    if ret and frame is not None:
        print("SUCCÈS : image récupérée !", frame.shape)
    else:
        print("ÉCHEC : caméra ouverte, mais aucune image.")

    camera.release()
