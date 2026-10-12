import sys
import os
import unittest
import numpy as np

# Permet d'importer le dossier src même en exécutant depuis le dossier tests
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.face_recognition import FaceRecognizer


class TestFaceRecognition(unittest.TestCase):

    def setUp(self):
        """S'exécute avant chaque test pour initialiser un environnement propre."""
        self.recognizer = FaceRecognizer(threshold=0.5)

    def test_empty_database_returns_unknown(self):
        """Vérifie que sans modèle chargé, le système renvoie bien 'Unknown'."""
        self.recognizer.known_encodings = []
        self.recognizer.known_names = []

        fake_face = np.random.rand(128)
        name, confidence = self.recognizer.recognize_face(fake_face)

        self.assertEqual(name, "Unknown")
        self.assertEqual(confidence, 0.0)

    def test_known_face_recognition(self):
        """Vérifie qu'un visage enregistré est reconnu avec une confiance élevée."""
        # On injecte manuellement une fausse signature connue
        dummy_signature = np.ones(128)
        self.recognizer.known_encodings = [dummy_signature]
        self.recognizer.known_names = ["Etudiant_Test"]

        # On teste avec un visage quasiment identique
        query_face = dummy_signature + np.random.normal(0, 0.01, 128)
        name, confidence = self.recognizer.recognize_face(query_face)

        self.assertEqual(name, "Etudiant_Test")
        self.assertGreater(confidence, 80.0)

    def test_unknown_rejection(self):
        """Vérifie qu'un visage très éloigné de la base est rejeté en 'Unknown'."""
        # Signature connue à +10
        self.recognizer.known_encodings = [np.ones(128) * 10.0]
        self.recognizer.known_names = ["Etudiant_1"]

        # Visage interrogé situé à -10 (distance énorme)
        query_face = np.ones(128) * -10.0
        name, confidence = self.recognizer.recognize_face(query_face)

        self.assertEqual(name, "Unknown")


if __name__ == "__main__":
    unittest.main()