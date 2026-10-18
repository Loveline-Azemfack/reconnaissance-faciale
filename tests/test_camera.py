import cv2
import numpy as np

from src.camera import display_result


def test_display_result_draws_rectangle():
    """Vérifie que display_result dessine un rectangle vert."""

    # Créer une image noire de 100 x 100 pixels
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # Simuler un résultat de détection
    faces = [
        {
            "box": (10, 10, 30, 30),
            "name": "Inconnu",
            "confidence": 45.0
        }
    ]

    # Appliquer l'affichage
    result = display_result(frame, faces)

    # Vérifier que la fonction renvoie une image
    assert result is frame

    # Vérifier que le coin du rectangle est vert (format BGR)
    assert np.array_equal(result[10, 10], [0, 255, 0])


def test_display_result_with_no_faces():
    """Vérifie que la fonction accepte une liste vide."""

    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    result = display_result(frame, [])

    assert result is frame