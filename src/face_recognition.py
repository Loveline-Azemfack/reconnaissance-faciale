import os
import pickle
import numpy as np
import face_recognition


class FaceRecognizer:
    def __init__(self, model_path="models/face_encodings.pkl", threshold=0.5):
        """
        Constructeur :
        - model_path : chemin vers le fichier contenant les encodages
        - threshold : distance maximale pour valider une identité (0.5 par défaut)
        """
        self.model_path = model_path
        self.threshold = threshold
        self.known_encodings = []
        self.known_names = []
        
        # Charge automatiquement la base à l'instanciation
        self.load_model()

    def load_model(self):
        """
        Charge les signatures (vecteurs 128D) et les noms depuis le fichier pickle.
        """
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                data = pickle.load(f)
            self.known_encodings = data.get("encodings", [])
            self.known_names = data.get("names", [])
            print(f"[FaceRecognizer] Modèle chargé avec succès : {len(self.known_names)} signatures trouvées.")
        else:
            print(f"[FaceRecognizer] Attention : aucun modèle trouvé à '{self.model_path}'.")

    def recognize_face(self, face_encoding):
        """
        Compare un vecteur de 128 dimensions avec la base de données.
        Retourne : (nom, confiance_en_pourcentage)
        """
        # Sécurité : si la base est vide
        if not self.known_encodings:
            return "Unknown", 0.0

        # 1. Calcul des distances euclidiennes avec tous les visages enregistrés
        distances = face_recognition.face_distance(self.known_encodings, face_encoding)

        # 2. Trouver l'écart le plus petit (la meilleure ressemblance)
        best_match_idx = np.argmin(distances)
        min_distance = distances[best_match_idx]

        # 3. Conversion de la distance en pourcentage de confiance (0% à 100%)
        confidence = max(0.0, min(100.0, round((1.0 - min_distance) * 100, 2)))

        # 4. Décision : connu ou inconnu selon le seuil
        if min_distance <= self.threshold:
            name = self.known_names[best_match_idx]
        else:
            name = "Unknown"

        return name, confidence

    def process_detected_faces(self, rgb_frame, face_locations):
        """
        Méthode d'intégration pour le groupe :
        Prend l'image RGB et la liste des coordonnées des visages détectés,
        puis renvoie une liste de tuples : [ (box, name, confidence), ... ]
        """
        results = []

        if not face_locations:
            return results

        # Extraction des 128 descripteurs pour chaque visage localisé
        encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for box, encoding in zip(face_locations, encodings):
            name, confidence = self.recognize_face(encoding)
            results.append((box, name, confidence))