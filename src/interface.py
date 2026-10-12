import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2


class FaceRecognitionInterface:
    def __init__(
        self,
        root,
        on_open_camera,
        on_close_camera,
        on_load_image,
        on_quit
    ):
        # On garde la fenêtre principale et les fonctions du contrôleur.
        # Comme ça, l'interface peut demander une action sans faire
        # elle-même tout le travail de reconnaissance.
        self.root = root

        self.on_open_camera = on_open_camera
        self.on_close_camera = on_close_camera
        self.on_load_image = on_load_image
        self.on_quit = on_quit

        # Tkinter doit conserver cette référence pour que l'image
        # reste visible dans la fenêtre.
        self.photo_reference = None

        self.root.title("Système de reconnaissance faciale")
        self.root.geometry("1100x750")
        self.root.minsize(800, 600)

        self._build_interface()

        # Si l'utilisateur ferme la fenêtre avec la croix,
        # on passe par la même fonction que le bouton Quitter.
        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.quit_application
        )

    def _build_interface(self):
        """Prépare les éléments visibles de l'application."""

        titre = tk.Label(
            self.root,
            text="SYSTÈME DE RECONNAISSANCE FACIALE",
            font=("Arial", 20, "bold")
        )
        titre.pack(pady=15)

        sous_titre = tk.Label(
            self.root,
            text="Identification des étudiants enregistrés",
            font=("Arial", 11)
        )
        sous_titre.pack(pady=5)

        # C'est ici que la vidéo de la webcam ou une photo apparaîtra.
        self.image_label = tk.Label(
            self.root,
            text="Aucune image affichée",
            bg="#202020",
            fg="white",
            font=("Arial", 14)
        )

        self.image_label.pack(
            padx=20,
            pady=15,
            fill="both",
            expand=True
        )

        # On regroupe les boutons pour garder une interface bien rangée.
        boutons = tk.Frame(self.root)
        boutons.pack(pady=10)

        self.camera_button = tk.Button(
            boutons,
            text="Ouvrir la webcam",
            width=20,
            command=self.open_camera
        )
        self.camera_button.grid(
            row=0,
            column=0,
            padx=5,
            pady=5
        )

        self.stop_button = tk.Button(
            boutons,
            text="Arrêter la webcam",
            width=20,
            command=self.close_camera
        )
        self.stop_button.grid(
            row=0,
            column=1,
            padx=5,
            pady=5
        )

        self.image_button = tk.Button(
            boutons,
            text="Charger une image",
            width=20,
            command=self.load_image
        )
        self.image_button.grid(
            row=0,
            column=2,
            padx=5,
            pady=5
        )

        self.quit_button = tk.Button(
            boutons,
            text="Quitter",
            width=20,
            command=self.quit_application
        )
        self.quit_button.grid(
            row=0,
            column=3,
            padx=5,
            pady=5
        )

        # Cette zone permet de savoir ce que fait l'application.
        self.status_label = tk.Label(
            self.root,
            text="Statut : prêt",
            font=("Arial", 11),
            anchor="w"
        )
        self.status_label.pack(
            fill="x",
            padx=20,
            pady=5
        )

        # Les noms des étudiants reconnus seront affichés ici.
        cadre_resultats = tk.LabelFrame(
            self.root,
            text="Résultats de la reconnaissance",
            padx=10,
            pady=5
        )
        cadre_resultats.pack(
            fill="x",
            padx=20,
            pady=10
        )

        self.results_label = tk.Label(
            cadre_resultats,
            text="Aucun visage reconnu pour le moment.",
            font=("Arial", 11),
            anchor="w",
            justify="left"
        )
        self.results_label.pack(fill="x")

    def show_image(self, image, status=None):
        """Affiche une image OpenCV dans la fenêtre."""

        if image is None:
            self.show_error("L'image reçue est vide.")
            return

        try:
            # OpenCV utilise généralement le format BGR,
            # alors que Pillow attend une image RGB.
            if len(image.shape) == 2:
                image_rgb = cv2.cvtColor(
                    image,
                    cv2.COLOR_GRAY2RGB
                )
            else:
                image_rgb = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB
                )

            image_pil = Image.fromarray(image_rgb)

            # On réduit l'image si elle est trop grande pour la fenêtre,
            # tout en conservant ses proportions.
            image_pil.thumbnail((1000, 420))

            self.photo_reference = ImageTk.PhotoImage(image_pil)

            self.image_label.configure(
                image=self.photo_reference,
                text=""
            )

            if status is not None:
                self.set_status(status)

        except (ValueError, cv2.error) as erreur:
            self.show_error(
                f"Erreur pendant l'affichage de l'image : {erreur}"
            )

    def show_results(self, results):
        """Affiche les personnes reconnues et leurs scores."""

        if not results:
            self.results_label.configure(
                text="Aucun visage détecté."
            )
            return

        lignes = []

        for resultat in results:
            nom = resultat.get("name", "Unknown")
            confiance = resultat.get("confidence")

            if confiance is None:
                lignes.append(str(nom))
            else:
                # On affiche le score reçu sans supposer qu'il est
                # forcément compris entre 0 et 1.
                lignes.append(
                    f"{nom} - Score : {confiance}"
                )

        self.results_label.configure(
            text="\n".join(lignes)
        )

    def set_status(self, message):
        """Actualise le petit message situé en bas de la fenêtre."""

        self.status_label.configure(
            text=f"Statut : {message}"
        )

    def open_camera(self):
        """Demande au contrôleur de démarrer la webcam."""

        try:
            self.on_open_camera()

        except Exception as erreur:
            self.show_error(
                f"Impossible d'ouvrir la webcam : {erreur}"
            )

    def close_camera(self):
        """Demande au contrôleur d'arrêter la webcam."""

        try:
            self.on_close_camera()

        except Exception as erreur:
            self.show_error(
                f"Impossible d'arrêter la webcam : {erreur}"
            )

    def load_image(self):
        """Permet de choisir une photo sur l'ordinateur."""

        chemin = filedialog.askopenfilename(
            parent=self.root,
            title="Choisir une image",
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.bmp"),
                ("Tous les fichiers", "*.*")
            ]
        )

        # Si l'utilisateur ferme la boîte de dialogue,
        # on ne lance aucun traitement.
        if not chemin:
            return

        try:
            self.on_load_image(chemin)

        except Exception as erreur:
            self.show_error(
                f"Impossible de traiter l'image : {erreur}"
            )

    def show_error(self, message):
        """Affiche un message lorsqu'une opération échoue."""

        self.set_status("Une erreur est survenue.")

        messagebox.showerror(
            "Erreur",
            message,
            parent=self.root
        )

    def quit_application(self):
        """Ferme l'application en passant par le contrôleur."""

        self.on_quit()