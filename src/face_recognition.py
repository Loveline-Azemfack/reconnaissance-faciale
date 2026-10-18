
import os
import pickle
import numpy as np


class FaceRecognizer:
    def __init__(self, model_path="models/face_encodings.pkl", threshold=0.5):
        self.model_path = model_path
        self.threshold = threshold
        self.known_encodings = []
        self.known_names = []
        self.load_model()

    def load_model(self):
        """Charge les signatures et les noms depuis le fichier pickle."""
        if not os.path.exists(self.model_path):
            print(
                f"[FaceRecognizer] Aucun modèle trouvé à "
                f"'{self.model_path}'."
            )
            return

        try:
            with open(self.model_path, "rb") as fichier:
                data = pickle.load(fichier)

            self.known_encodings = data.get("encodings", [])
            self.known_names = data.get("names", [])

            print(
                "[FaceRecognizer] Modèle chargé : "
                f"{len(self.known_names)} signatures."
            )
        except (OSError, pickle.UnpicklingError, EOFError, AttributeError, ValueError) as erreur:
            print(f"[FaceRecognizer] Erreur de chargement : {erreur}")

    def recognize_face(self, face_encoding):
        """Compare une signature à la base et renvoie (nom, confiance %)."""
        if not self.known_encodings:
            return "Unknown", 0.0

        try:
            signature = np.asarray(face_encoding, dtype=np.float32).ravel()
            signatures_connues = np.asarray(
                self.known_encodings, dtype=np.float32
            )

            if (
                signatures_connues.ndim != 2
                or signatures_connues.shape[1] != signature.size
                or not np.all(np.isfinite(signature))
                or not np.all(np.isfinite(signatures_connues))
            ):
                return "Unknown", 0.0

            distances = np.linalg.norm(
                signatures_connues - signature, axis=1
            )
            meilleur_index = int(np.argmin(distances))
            distance_min = float(distances[meilleur_index])

            confiance = max(
                0.0,
                min(100.0, round((1.0 - distance_min) * 100, 2))
            )

            if distance_min <= self.threshold:
                nom = self.known_names[meilleur_index]
            else:
                nom = "Unknown"

            return nom, confiance

        except (TypeError, ValueError, IndexError):
            return "Unknown", 0.0

    def process_detected_faces(self, rgb_frame, face_locations):
        """
        Préserve l'interface attendue par le projet.
        Les signatures doivent être fournies par le détecteur utilisé.
        """
        results = []

        if not face_locations:
            return results

        # Cette méthode nécessite des signatures déjà extraites.
        # L'extraction dépend du modèle de détection utilisé.
        return results
