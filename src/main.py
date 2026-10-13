import tkinter as tk
import importlib
import cv2

from src.interface import FaceRecognitionInterface


class ApplicationController:
    def __init__(self, root, camera_module=None):
        self.root = root

        # On pourra utiliser le vrai module de caméra dans l'application
        # et un faux module pendant les tests.
        self.camera = camera_module

        self.camera_active = False
        self.camera_callback_id = None
        self.closing = False

        # L'interface reçoit les fonctions qu'elle doit appeler.
        self.interface = FaceRecognitionInterface(
            root=self.root,
            on_open_camera=self.start_camera,
            on_close_camera=self.stop_camera,
            on_load_image=self.load_image,
            on_quit=self.quit_application
        )

    def _get_camera_module(self):
        """Charge le module de webcam lorsqu'on en a besoin."""

        if self.camera is None:
            self.camera = importlib.import_module("src.camera")

        return self.camera

    def start_camera(self):
        """Ouvre la webcam et lance le traitement des images."""

        if self.camera_active or self.closing:
            return

        camera = self._get_camera_module()

        # On vérifie d'abord que la caméra a bien été ouverte.
        ouverte = camera.open_camera()

        if not ouverte:
            self.interface.set_status(
                "Impossible d'ouvrir la webcam."
            )
            return

        self.camera_active = True

        self.interface.set_status("Webcam active.")

        # On traite les images progressivement pour que la fenêtre
        # reste réactive pendant l'utilisation de la webcam.
        self._update_camera_frame()

    def _update_camera_frame(self):
        """Récupère, traite et affiche une image de la webcam."""

        if not self.camera_active or self.closing:
            return

        try:
            camera = self._get_camera_module()

            # On récupère l'image actuelle de la caméra.
            frame = camera.capture_frame()

            if frame is None:
                self.interface.set_status(
                    "Impossible de lire l'image de la webcam."
                )
                self.stop_camera()
                return

            # Le module de traitement doit renvoyer les résultats
            # de détection et de reconnaissance.
            results = camera.process_frame(frame)

            # On dessine les résultats sur une copie pour conserver
            # l'image originale si on en a encore besoin.
            image_annotee = camera.draw_results(
                frame.copy(),
                results
            )

            if image_annotee is None:
                image_annotee = frame

            self.interface.show_image(
                image_annotee,
                status="Reconnaissance en cours..."
            )

            self.interface.show_results(results)

            # Le prochain traitement sera lancé après un court délai.
            # Cela évite de bloquer la fenêtre Tkinter.
            self.camera_callback_id = self.root.after(
                30,
                self._update_camera_frame
            )

        except Exception as erreur:
            self.stop_camera()

            self.interface.show_error(
                f"Erreur pendant le traitement vidéo : {erreur}"
            )

    def stop_camera(self):
        """Arrête la capture et libère les ressources utilisées."""

        self.camera_active = False

        # On annule le prochain traitement s'il était programmé.
        if self.camera_callback_id is not None:
            try:
                self.root.after_cancel(
                    self.camera_callback_id
                )
            except tk.TclError:
                pass

            self.camera_callback_id = None

        # On ne ferme la caméra que si le module a été chargé.
        if self.camera is not None:
            try:
                self.camera.close_camera()
            except Exception as erreur:
                print(
                    f"Erreur lors de la fermeture de la caméra : {erreur}"
                )

        if not self.closing:
            self.interface.set_status("Webcam arrêtée.")

    def load_image(self, chemin):
        """Charge une image et l'envoie au traitement."""

        image = cv2.imread(chemin)

        if image is None:
            self.interface.show_error(
                "Impossible de lire l'image sélectionnée."
            )
            return

        camera = self._get_camera_module()

        # Le même traitement est réutilisé pour les photos et la vidéo.
        resultats = camera.process_frame(image)

        image_annotee = camera.draw_results(
            image.copy(),
            resultats
        )

        if image_annotee is None:
            image_annotee = image

        self.interface.show_image(
            image_annotee,
            status="Image traitée."
        )

        self.interface.show_results(resultats)

    def quit_application(self):
        """Arrête les opérations en cours avant de fermer la fenêtre."""

        if self.closing:
            return

        self.closing = True

        try:
            self.stop_camera()
        finally:
            self.root.destroy()


def main():
    """Lance l'application."""

    root = tk.Tk()

    ApplicationController(root)

    root.mainloop()
    
if __name__ == "__main__":
    main()