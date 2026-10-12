
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

        # Palette de couleurs noir et violet.
        self.colors = {
            "background": "#0B0B10",
            "panel": "#171720",
            "purple": "#8B5CF6",
            "purple_hover": "#7042D8",
            "purple_light": "#A78BFA",
            "text": "#F5F3FF",
            "muted": "#B8B5C8",
            "border": "#343044",
            "image_background": "#101016",
            "red": "#E5484D",
            "red_hover": "#BE3439",
        }

        # Police moderne disponible sur la plupart des systèmes.
        self.font_title = ("Segoe UI", 21, "bold")
        self.font_subtitle = ("Segoe UI", 11)
        self.font_normal = ("Segoe UI", 11)
        self.font_button = ("Segoe UI", 10, "bold")
        self.font_results = ("Segoe UI", 11)

        self.root.title("Système de reconnaissance faciale")
        self.root.geometry("1100x750")
        self.root.minsize(800, 600)
        self.root.configure(bg=self.colors["background"])

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
            font=self.font_title,
            bg=self.colors["background"],
            fg=self.colors["purple_light"]
        )
        titre.pack(pady=(18, 8))

        sous_titre = tk.Label(
            self.root,
            text="Identification des étudiants enregistrés",
            font=self.font_subtitle,
            bg=self.colors["background"],
            fg=self.colors["muted"]
        )
        sous_titre.pack(pady=5)

        # C'est ici que la vidéo de la webcam ou une photo apparaîtra.
        self.image_label = tk.Label(
            self.root,
            text="Aucune image affichée",
            bg=self.colors["image_background"],
            fg=self.colors["muted"],
            font=("Segoe UI", 14),
            bd=0,
            highlightthickness=1,
            highlightbackground=self.colors["border"]
        )

        self.image_label.pack(
            padx=20,
            pady=15,
            fill="both",
            expand=True
        )

        # On regroupe les boutons pour garder une interface bien rangée.
        boutons = tk.Frame(
            self.root,
            bg=self.colors["background"]
        )
        boutons.pack(pady=10)

        self.camera_button = self._create_rounded_button(
            boutons,
            text="Ouvrir la webcam",
            command=self.open_camera,
            width=160,
            color=self.colors["purple"],
            hover_color=self.colors["purple_hover"]
        )
        self.camera_button.grid(
            row=0,
            column=0,
            padx=5,
            pady=5
        )

        self.stop_button = self._create_rounded_button(
            boutons,
            text="Arrêter la webcam",
            command=self.close_camera,
            width=160,
            color=self.colors["panel"],
            hover_color="#302A40"
        )
        self.stop_button.grid(
            row=0,
            column=1,
            padx=5,
            pady=5
        )

        self.image_button = self._create_rounded_button(
            boutons,
            text="Charger une image",
            command=self.load_image,
            width=160,
            color=self.colors["panel"],
            hover_color="#302A40"
        )
        self.image_button.grid(
            row=0,
            column=2,
            padx=5,
            pady=5
        )

        self.quit_button = self._create_rounded_button(
            boutons,
            text="Quitter",
            command=self.quit_application,
            width=160,
            color="#48202A",
            hover_color="#672A38",
            text_color="#FFC4CC"
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
            font=self.font_normal,
            bg=self.colors["background"],
            fg=self.colors["purple_light"],
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
            pady=5,
            font=("Segoe UI", 11, "bold"),
            bg=self.colors["panel"],
            fg=self.colors["purple_light"],
            bd=1,
            relief="solid",
            highlightbackground=self.colors["border"]
        )
        cadre_resultats.pack(
            fill="x",
            padx=20,
            pady=10
        )

        self.results_label = tk.Label(
            cadre_resultats,
            text="Aucun visage reconnu pour le moment.",
            font=self.font_results,
            bg=self.colors["panel"],
            fg=self.colors["text"],
            anchor="w",
            justify="left"
        )
        self.results_label.pack(fill="x")

    def _create_rounded_button(
        self,
        parent,
        text,
        command,
        width=160,
        height=42,
        color=None,
        hover_color=None,
        text_color=None,
        radius=14
    ):
        """
        Crée un bouton visuellement arrondi avec Canvas Tkinter.

        Le bouton conserve une commande classique et sa position
        dans la grille reste définie par les appels à grid().
        """

        color = color or self.colors["purple"]
        hover_color = hover_color or self.colors["purple_hover"]
        text_color = text_color or self.colors["text"]

        canvas = tk.Canvas(
            parent,
            width=width,
            height=height,
            bg=self.colors["background"],
            highlightthickness=0,
            bd=0,
            cursor="hand2"
        )

        def rounded_rectangle(x1, y1, x2, y2, r, fill):
            """Dessine un rectangle aux coins arrondis."""

            points = [
                x1 + r, y1,
                x2 - r, y1,
                x2, y1,
                x2, y1 + r,
                x2, y2 - r,
                x2, y2,
                x2 - r, y2,
                x1 + r, y2,
                x1, y2,
                x1, y2 - r,
                x1, y1 + r,
                x1, y1
            ]

            return canvas.create_polygon(
                points,
                smooth=True,
                splinesteps=24,
                fill=fill,
                outline=fill
            )

        shape = rounded_rectangle(
            2, 2, width - 2, height - 2,
            radius, color
        )

        label = canvas.create_text(
            width // 2,
            height // 2,
            text=text,
            font=self.font_button,
            fill=text_color
        )

        def on_enter(event):
            canvas.itemconfigure(shape, fill=hover_color)
            canvas.itemconfigure(shape, outline=hover_color)

        def on_leave(event):
            canvas.itemconfigure(shape, fill=color)
            canvas.itemconfigure(shape, outline=color)

        def on_click(event):
            command()

        canvas.bind("<Enter>", on_enter)
        canvas.bind("<Leave>", on_leave)
        canvas.bind("<Button-1>", on_click)

        canvas.tag_bind(shape, "<Enter>", on_enter)
        canvas.tag_bind(shape, "<Leave>", on_leave)
        canvas.tag_bind(shape, "<Button-1>", on_click)

        canvas.tag_bind(label, "<Enter>", on_enter)
        canvas.tag_bind(label, "<Leave>", on_leave)
        canvas.tag_bind(label, "<Button-1>", on_click)

        return canvas

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
