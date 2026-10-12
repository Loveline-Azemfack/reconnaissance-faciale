import cv2
import csv
import json
import socket
import threading
import webbrowser

from datetime import datetime
from pathlib import Path

import numpy as np
from flask import Flask, request, render_template_string, redirect, url_for
from insightface.app import FaceAnalysis


# ============================================================
# 1. CONFIGURATION DU PROJET
# ============================================================

DOSSIER = Path(__file__).resolve().parent

PERSONNES = DOSSIER / "personnes"
PHOTOS = DOSSIER / "photos_a_analyser"
RESULTATS = DOSSIER / "resultats"
BASE = DOSSIER / "base_de_donnees"

FICHIER_VISAGES = BASE / "encodages.json"
FICHIER_PRESENCES = BASE / "presences.csv"

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Seuil initial de similarité cosinus.
# Il devra être évalué sur tes propres photos.
SEUIL_RECONNAISSANCE = 0.45

# Délai avant de pouvoir enregistrer une nouvelle présence
# de la même personne.
DELAI_PRESENCE_MINUTES = 30

PORT_TELEPHONE = 5000


for dossier in (PERSONNES, PHOTOS, RESULTATS, BASE):
    dossier.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CHARGEMENT DU MODELE DE DETECTION
# ============================================================

print("\nChargement du modèle de reconnaissance faciale...")
print("Le premier lancement peut télécharger des modèles.")

app_visage = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app_visage.prepare(
    ctx_id=0,
    det_size=(640, 640)
)

print("Modèle chargé avec succès !")


# ============================================================
# 3. CHARGEMENT ET SAUVEGARDE DES PERSONNES
# ============================================================

def charger_personnes():
    """Charge les caractéristiques faciales enregistrées."""

    if not FICHIER_VISAGES.exists():
        return {}

    try:
        with open(FICHIER_VISAGES, "r", encoding="utf-8") as fichier:
            donnees = json.load(fichier)

        return donnees

    except (OSError, json.JSONDecodeError):
        print("Attention : impossible de lire la base des visages.")
        return {}


def sauvegarder_personnes(donnees):
    """Enregistre les caractéristiques faciales."""

    with open(FICHIER_VISAGES, "w", encoding="utf-8") as fichier:
        json.dump(donnees, fichier, ensure_ascii=False, indent=4)


# ============================================================
# 4. EXTRACTION DES CARACTERISTIQUES FACIALES
# ============================================================

def extraire_visages(image):
    """Retourne les visages détectés dans une image."""

    return app_visage.get(image)


def comparer_visage(embeddings, visage):
    """
    Compare un visage aux personnes enregistrées.

    Retourne le nom le plus proche et sa similarité.
    Une similarité élevée ne garantit pas l'identité.
    """

    if not embeddings:
        return "Inconnu", 0.0

    caracteristiques = getattr(visage, "normed_embedding", None)

    if caracteristiques is None:
        caracteristiques = visage.embedding

    caracteristiques = np.asarray(caracteristiques, dtype=np.float32)
    norme = np.linalg.norm(caracteristiques)

    if norme == 0:
        return "Inconnu", 0.0

    caracteristiques = caracteristiques / norme

    meilleur_nom = "Inconnu"
    meilleure_similarite = -1.0

    for nom, photos in embeddings.items():
        for photo in photos:

            reference = np.asarray(photo, dtype=np.float32)
            norme_reference = np.linalg.norm(reference)

            if norme_reference == 0:
                continue

            reference = reference / norme_reference

            similarite = float(np.dot(caracteristiques, reference))

            if similarite > meilleure_similarite:
                meilleure_similarite = similarite
                meilleur_nom = nom

    if meilleure_similarite < SEUIL_RECONNAISSANCE:
        return "Inconnu", meilleure_similarite

    return meilleur_nom, meilleure_similarite


# ============================================================
# 5. ENREGISTREMENT D'UNE PERSONNE
# ============================================================

def enregistrer_personnes():
    """
    Chaque personne doit avoir son propre dossier.

    Exemple :
        personnes/
            Gabriel/
                photo1.jpg
                photo2.jpg
    """

    donnees = charger_personnes()
    nombre_personnes = 0

    dossiers = [p for p in PERSONNES.iterdir() if p.is_dir()]

    if not dossiers:
        print("\nAucun dossier de personne trouvé.")
        print("Crée un dossier par personne dans :", PERSONNES)
        return

    for dossier_personne in dossiers:

        nom = dossier_personne.name
        caracteristiques = []

        photos = [
            p for p in dossier_personne.iterdir()
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        ]

        for photo in photos:

            image = cv2.imread(str(photo))

            if image is None:
                print("Photo illisible :", photo.name)
                continue

            visages = extraire_visages(image)

            if len(visages) == 0:
                print("Aucun visage détecté dans :", photo.name)
                continue

            if len(visages) > 1:
                print(
                    "Plusieurs visages dans", photo.name,
                    "- photo ignorée pour éviter une mauvaise association."
                )
                continue

            visage = visages[0]

            caracteristiques.append(
                visage.normed_embedding.astype(float).tolist()
            )

            print("Photo de référence enregistrée :", nom, "-", photo.name)

        if caracteristiques:
            donnees[nom] = caracteristiques
            nombre_personnes += 1
        else:
            print("Aucune photo de référence exploitable pour :", nom)

    sauvegarder_personnes(donnees)

    print("\nEnregistrement terminé.")
    print("Personnes disposant de références :", nombre_personnes)


# ============================================================
# 6. GESTION DES PRESENCES
# ============================================================

def initialiser_fichier_presences():

    if not FICHIER_PRESENCES.exists():

        with open(
            FICHIER_PRESENCES,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as fichier:

            ecrivain = csv.writer(fichier, delimiter=";")

            ecrivain.writerow([
                "Nom",
                "Date",
                "Heure",
                "Similarite"
            ])


def enregistrer_presence(nom, similarite):

    if nom == "Inconnu":
        return

    initialiser_fichier_presences()

    maintenant = datetime.now()
    date_aujourdhui = maintenant.strftime("%Y-%m-%d")

    # Vérifier les dernières présences de cette personne.
    with open(
        FICHIER_PRESENCES,
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as fichier:

        lignes = list(csv.DictReader(fichier, delimiter=";"))

    for ligne in reversed(lignes):

        if ligne["Nom"] != nom:
            continue

        if ligne["Date"] != date_aujourdhui:
            break

        try:
            heure_precedente = datetime.strptime(
                ligne["Date"] + " " + ligne["Heure"],
                "%Y-%m-%d %H:%M:%S"
            )

            ecart = (maintenant - heure_precedente).total_seconds() / 60

            if ecart < DELAI_PRESENCE_MINUTES:
                print("Présence déjà enregistrée récemment pour :", nom)
                return

        except ValueError:
            continue

        break

    with open(
        FICHIER_PRESENCES,
        "a",
        newline="",
        encoding="utf-8-sig"
    ) as fichier:

        ecrivain = csv.writer(fichier, delimiter=";")

        ecrivain.writerow([
            nom,
            date_aujourdhui,
            maintenant.strftime("%H:%M:%S"),
            f"{similarite:.3f}"
        ])

    print("Présence enregistrée :", nom, maintenant.strftime("%H:%M:%S"))


def consulter_presences():

    initialiser_fichier_presences()

    with open(
        FICHIER_PRESENCES,
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as fichier:

        lecteur = csv.reader(fichier, delimiter=";")

        for ligne in lecteur:
            print(" | ".join(ligne))


# ============================================================
# 7. ANALYSE D'UNE PHOTO
# ============================================================

def analyser_photo(chemin_photo, afficher=True):

    image = cv2.imread(str(chemin_photo))

    if image is None:
        print("Impossible de lire :", chemin_photo)
        return

    donnees = charger_personnes()
    visages = extraire_visages(image)

    print("\nPhoto :", chemin_photo.name)
    print("Nombre de visages détectés :", len(visages))

    for numero, visage in enumerate(visages, start=1):

        x1, y1, x2, y2 = visage.bbox.astype(int)

        nom, similarite = comparer_visage(donnees, visage)

        if nom == "Inconnu":
            couleur = (0, 0, 255)
            etiquette = "Inconnu"
        else:
            couleur = (0, 255, 0)
            etiquette = nom

            # Une reconnaissance sur une photo ne constitue pas
            # à elle seule une preuve suffisante de présence.
            print(
                "Correspondance possible :", nom,
                "- similarité :", round(similarite, 3)
            )

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            couleur,
            2
        )

        cv2.putText(
            image,
            etiquette,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            couleur,
            2
        )

        print(
            f"Visage {numero} : {nom} "
            f"(similarité : {similarite:.3f})"
        )

    destination = RESULTATS / f"{chemin_photo.stem}_analyse.jpg"

    cv2.imwrite(str(destination), image)

    print("Résultat enregistré :", destination)

    if afficher:
        cv2.imshow("Analyse des visages", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def analyser_toutes_les_photos():

    photos = [
        p for p in PHOTOS.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    ]

    if not photos:
        print("\nAucune photo dans :", PHOTOS)
        return

    for photo in photos:
        analyser_photo(photo)


# ============================================================
# 8. RECONNAISSANCE AVEC LA WEBCAM
# ============================================================

def utiliser_webcam():

    donnees = charger_personnes()

    if not donnees:
        print("Enregistre d'abord les personnes de référence.")
        return

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Impossible d'ouvrir la webcam.")
        return

    print("\nWebcam démarrée.")
    print("Appuie sur Q pour quitter.")
    print("Appuie sur P pour demander l'enregistrement d'une présence.")

    # L'enregistrement des présences nécessite une action explicite.
    demander_presence = False

    while True:

        ok, image = camera.read()

        if not ok:
            print("Impossible de lire l'image de la webcam.")
            break

        visages = extraire_visages(image)

        for visage in visages:

            x1, y1, x2, y2 = visage.bbox.astype(int)

            nom, similarite = comparer_visage(donnees, visage)

            if nom == "Inconnu":
                couleur = (0, 0, 255)
                etiquette = "Inconnu"
            else:
                couleur = (0, 255, 0)
                etiquette = f"{nom} ({similarite:.2f})"

                if demander_presence:
                    enregistrer_presence(nom, similarite)

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                couleur,
                2
            )

            cv2.putText(
                image,
                etiquette,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                couleur,
                2
            )

        cv2.imshow("Reconnaissance faciale - Webcam", image)

        touche = cv2.waitKey(1) & 0xFF

        if touche == ord("q"):
            break

        if touche == ord("p"):
            demander_presence = True
            print("Mode présence activé pour la prochaine image.")

    camera.release()
    cv2.destroyAllWindows()


# ============================================================
# 9. SERVEUR WEB POUR LE TELEPHONE
# ============================================================

serveur = Flask(__name__)


PAGE_TELEPHONE = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Reconnaissance faciale</title>
</head>
<body style="font-family:Arial;max-width:600px;margin:30px auto;padding:15px">

    <h1>Analyse d'une photo</h1>

    <p>
        Choisis une photo ou utilise la caméra de ton téléphone.
    </p>

    <form action="/envoyer" method="post" enctype="multipart/form-data">

        <input
            type="file"
            name="photo"
            accept="image/*"
            capture="environment"
            required
        >

        <br><br>

        <button type="submit">Envoyer et analyser</button>

    </form>

    <p>
        Les résultats seront enregistrés sur l'ordinateur.
    </p>

</body>
</html>
"""


@serveur.route("/", methods=["GET"])
def page_accueil():
    return render_template_string(PAGE_TELEPHONE)


@serveur.route("/envoyer", methods=["POST"])
def recevoir_photo():

    fichier = request.files.get("photo")

    if fichier is None or fichier.filename == "":
        return "Aucune photo reçue.", 400

    extension = Path(fichier.filename).suffix.lower()

    if extension not in EXTENSIONS:
        return "Format non pris en charge.", 400

    # Nom de fichier créé par le serveur pour éviter
    # d'utiliser directement le nom fourni par le navigateur.
    nom_fichier = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination = PHOTOS / f"telephone_{nom_fichier}{extension}"

    fichier.save(destination)

    try:
        analyser_photo(destination, afficher=False)

        return (
            "<h2>Analyse terminée</h2>"
            "<p>La photo a été analysée.</p>"
            "<p>Les résultats sont enregistrés sur l'ordinateur.</p>"
            '<p><a href="/">Analyser une autre photo</a></p>'
        )

    except Exception as erreur:
        print("Erreur pendant l'analyse :", erreur)

        return (
            "La photo a été reçue, mais l'analyse a échoué. "
            "Consulte le terminal de l'ordinateur."
        ), 500


def lancer_serveur_telephone():

    print("\nServeur du téléphone en préparation...")

    try:
        nom_hote = socket.gethostname()
        adresse_ip = socket.gethostbyname(nom_hote)

        print("\nSur l'ordinateur : http://127.0.0.1:" + str(PORT_TELEPHONE))
        print(
            "Sur le téléphone : http://"
            + adresse_ip
            + ":"
            + str(PORT_TELEPHONE)
        )

        print(
            "\nLe téléphone et l'ordinateur doivent être sur le même réseau."
        )
        print("Arrête le serveur avec Ctrl+C.")

    except OSError:
        print("Impossible de déterminer l'adresse IP locale.")

    serveur.run(
        host="0.0.0.0",
        port=PORT_TELEPHONE,
        debug=False,
        use_reloader=False
    )


# ============================================================
# 10. MENU PRINCIPAL
# ============================================================

def menu():

    while True:

        print("\n========================================")
        print("     APPLICATION DE RECONNAISSANCE")
        print("========================================")

        print("1. Enregistrer les personnes de référence")
        print("2. Analyser les photos du dossier")
        print("3. Utiliser la webcam")
        print("4. Recevoir une photo depuis le téléphone")
        print("5. Consulter les présences")
        print("0. Quitter")

        choix = input("\nTon choix : ").strip()

        if choix == "1":
            enregistrer_personnes()

        elif choix == "2":
            analyser_toutes_les_photos()

        elif choix == "3":
            utiliser_webcam()

        elif choix == "4":
            lancer_serveur_telephone()

        elif choix == "5":
            consulter_presences()

        elif choix == "0":
            print("Fermeture du programme.")
            break

        else:
            print("Choix invalide. Réessaie.")


if __name__ == "__main__":

    initialiser_fichier_presences()
    menu()

