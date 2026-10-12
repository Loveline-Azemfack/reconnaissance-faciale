import numpy as np
import pytest
from unittest.mock import Mock

from src.main import ApplicationController


class FakeRoot:
    """Petite fenêtre simulée pour éviter d'ouvrir Tkinter pendant les tests."""

    def __init__(self):
        self.callbacks = {}
        self.next_id = 1
        self.destroyed = False

    def after(self, delay, callback):
        # On garde le rappel en mémoire, sans l'exécuter immédiatement.
        # Cela permet de tester le contrôleur sans lancer une vraie boucle.
        callback_id = f"after_{self.next_id}"
        self.next_id += 1

        self.callbacks[callback_id] = callback

        return callback_id

    def after_cancel(self, callback_id):
        self.callbacks.pop(callback_id, None)

    def protocol(self, *args):
        pass

    def destroy(self):
        self.destroyed = True


class FakeInterface:
    """Remplace l'interface graphique pendant les tests."""

    def __init__(
        self,
        root,
        on_open_camera,
        on_close_camera,
        on_load_image,
        on_quit
    ):
        self.root = root

        self.on_open_camera = on_open_camera
        self.on_close_camera = on_close_camera
        self.on_load_image = on_load_image
        self.on_quit = on_quit

        self.images = []
        self.results = []
        self.status = ""
        self.errors = []

    def show_image(self, image, status=None):
        self.images.append(image)

        if status:
            self.status = status

    def show_results(self, results):
        self.results = results

    def set_status(self, message):
        self.status = message

    def show_error(self, message):
        self.errors.append(message)


@pytest.fixture
def fake_camera():
    """Prépare une caméra fictive avec des résultats connus."""

    camera = Mock()

    camera.open_camera.return_value = True

    # Une image noire suffit pour tester le passage des images.
    camera.capture_frame.return_value = np.zeros(
        (100, 100, 3),
        dtype=np.uint8
    )

    camera.process_frame.return_value = [
        {
            "bbox": (10, 10, 40, 40),
            "name": "Etudiant1",
            "confidence": 0.92
        }
    ]

    # Pour ce test, on renvoie simplement l'image reçue.
    camera.draw_results.side_effect = (
        lambda image, results: image
    )

    return camera


@pytest.fixture
def controller(monkeypatch, fake_camera):
    """Prépare le contrôleur sans ouvrir une vraie fenêtre ni caméra."""

    # On importe le module principal de l'application.
    import src.main

    # On remplace l'interface utilisée par main.py
    # par notre fausse interface réservée aux tests.
    monkeypatch.setattr(
        src.main,
        "FaceRecognitionInterface",
        FakeInterface
    )

    # On crée une fausse fenêtre pour éviter d'ouvrir Tkinter.
    root = FakeRoot()

    # On construit le contrôleur avec la caméra simulée.
    app = ApplicationController(
        root,
        camera_module=fake_camera
    )

    return app, root, fake_camera


def test_camera_starts(controller):
    """Vérifie que le démarrage ouvre bien la caméra."""

    app, root, camera = controller

    app.start_camera()

    camera.open_camera.assert_called_once()

    assert app.camera_active is True


def test_camera_frame_is_processed(controller):
    """Vérifie qu'une image est capturée, traitée et affichée."""

    app, root, camera = controller

    app.start_camera()

    camera.capture_frame.assert_called_once()
    camera.process_frame.assert_called_once()
    camera.draw_results.assert_called_once()

    assert len(app.interface.images) == 1

    assert app.interface.results[0]["name"] == "Etudiant1"


def test_camera_stops(controller):
    """Vérifie que l'arrêt libère la caméra."""

    app, root, camera = controller

    app.start_camera()
    app.stop_camera()

    camera.close_camera.assert_called_once()

    assert app.camera_active is False


def test_camera_open_failure(controller):
    """Vérifie le comportement si la caméra ne peut pas démarrer."""

    app, root, camera = controller

    camera.open_camera.return_value = False

    app.start_camera()

    assert app.camera_active is False

    # Aucune image ne doit être capturée si l'ouverture a échoué.
    camera.capture_frame.assert_not_called()


def test_invalid_image_is_rejected(controller, monkeypatch):
    """Vérifie qu'une image illisible est signalée."""

    app, root, camera = controller

    import src.main

    # On simule une image que OpenCV ne parvient pas à lire.
    monkeypatch.setattr(
        src.main.cv2,
        "imread",
        lambda chemin: None
    )

    app.load_image("image_inexistante.jpg")

    assert len(app.interface.errors) == 1

    camera.process_frame.assert_not_called()


def test_valid_image_is_processed(controller, monkeypatch):
    """Vérifie qu'une image valide est transmise au traitement."""

    app, root, camera = controller

    import src.main

    image = np.zeros(
        (100, 100, 3),
        dtype=np.uint8
    )

    monkeypatch.setattr(
        src.main.cv2,
        "imread",
        lambda chemin: image
    )

    app.load_image("photo.jpg")

    camera.process_frame.assert_called_once()
    camera.draw_results.assert_called_once()

    assert len(app.interface.images) == 1


def test_quit_closes_application(controller):
    """Vérifie que la fermeture arrête la caméra et ferme la fenêtre."""

    app, root, camera = controller

    app.start_camera()

    app.quit_application()

    camera.close_camera.assert_called_once()

    assert root.destroyed is True