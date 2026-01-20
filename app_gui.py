import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import transforms, models
import os
import sys

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = 'modele.pth'
DATA_DIR = './data_avions/train'
IMG_SIZE = 224
DISPLAY_SIZE = (400, 300)  # Taille d'affichage de l'image dans l'app


class PlaneClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Military Aircraft Recognizer")
        self.root.geometry("800x600")
        self.root.resizable(False, False)

        # --- 1. Initialisation du système ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = self._get_class_names()
        self.model = self._load_model()
        self.transform = self._get_transforms()

        # --- 2. Construction de l'interface ---
        self._setup_ui()

    def _get_class_names(self):
        """Récupère les noms des classes (dossiers ou liste manuelle)"""
        if os.path.exists(DATA_DIR):
            return sorted(os.listdir(DATA_DIR))
        else:
            # Fallback si le dossier d'entrainement n'est pas présent sur la machine de l'utilisateur
            return ['F-35 fighter jet', 'Mirage fighter jet', 'Rafale fighter jet', 'Soukhoï Su-35 fighter jet']

    def _get_transforms(self):
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _load_model(self):
        """Charge le modèle PyTorch"""
        try:
            print(f"Chargement du modèle sur {self.device}...")
            model = models.resnet18(weights=None)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, len(self.class_names))

            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Le fichier modèle '{MODEL_PATH}' est introuvable.")

            model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            model = model.to(self.device)
            model.eval()
            print("Modèle chargé avec succès.")
            return model
        except Exception as e:
            messagebox.showerror("Erreur Fatale", f"Impossible de charger le modèle :\n{e}")
            sys.exit(1)

    def _setup_ui(self):
        # Style
        style = ttk.Style()
        style.theme_use('clam')

        # Titre
        title_frame = tk.Frame(self.root, bg="#2c3e50", height=80)
        title_frame.pack(fill=tk.X)
        lbl_title = tk.Label(title_frame, text="Reconnaissance d'aéronef militaire par IA", font=("Helvetica", 24, "bold"), bg="#2c3e50",
                             fg="white")
        lbl_title.pack(pady=20)

        # Conteneur Principal
        main_frame = tk.Frame(self.root, bg="#ecf0f1")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # -- GAUCHE : Image --
        left_frame = tk.Frame(main_frame, bg="white", bd=2, relief=tk.RIDGE)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.lbl_image = tk.Label(left_frame, text="Aucune image sélectionnée", bg="#bdc3c7", fg="#7f8c8d")
        self.lbl_image.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # -- DROITE : Résultats --
        right_frame = tk.Frame(main_frame, bg="#ecf0f1", width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        right_frame.pack_propagate(False)  # Force la largeur fixe

        # Bouton Charger
        btn_load = tk.Button(right_frame, text="📂 Charger une Photo", command=self.load_image,
                             font=("Helvetica", 14), bg="#3498db", fg="white", activebackground="#2980b9",
                             relief=tk.FLAT, pady=10)
        btn_load.pack(fill=tk.X, pady=(0, 20))

        # Résultat Principal
        lbl_result_title = tk.Label(right_frame, text="Prédiction :", font=("Helvetica", 12), bg="#ecf0f1",
                                    fg="#7f8c8d")
        lbl_result_title.pack(anchor="w")

        self.lbl_pred_class = tk.Label(right_frame, text="...", font=("Helvetica", 18, "bold"), bg="#ecf0f1",
                                       fg="#2c3e50", wraplength=280)
        self.lbl_pred_class.pack(anchor="w", pady=(0, 5))

        self.lbl_confidence = tk.Label(right_frame, text="Confiance : -", font=("Helvetica", 12), bg="#ecf0f1",
                                       fg="#7f8c8d")
        self.lbl_confidence.pack(anchor="w", pady=(0, 20))

        # Détails (Barres de probabilités)
        tk.Label(right_frame, text="Détails de l'analyse :", font=("Helvetica", 12, "bold"), bg="#ecf0f1").pack(
            anchor="w", pady=(10, 5))

        self.stats_text = tk.Text(right_frame, height=10, bg="#ecf0f1", bd=0, font=("Consolas", 10), state=tk.DISABLED)
        self.stats_text.pack(fill=tk.X)

        # Footer
        tk.Label(self.root, text="Propulsé par PyTorch & ResNet18", bg="#ecf0f1", fg="#95a5a6", font=("Arial", 8)).pack(
            side=tk.BOTTOM, pady=5)

    def load_image(self):
        file_path = filedialog.askopenfilename(title="Choisir une image",
                                               filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
        if not file_path:
            return

        # 1. Affichage de l'image
        try:
            pil_img = Image.open(file_path).convert('RGB')

            # Redimensionnement pour l'affichage (Garder le ratio)
            pil_img.thumbnail(DISPLAY_SIZE, Image.LANCZOS)
            self.tk_img = ImageTk.PhotoImage(pil_img)

            self.lbl_image.config(image=self.tk_img, text="")  # Enlever le texte par défaut

            # 2. Lancer la prédiction
            self.predict(file_path)

        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible d'ouvrir l'image.\n{e}")

    def predict(self, file_path):
        # Préparation pour PyTorch
        img_pil = Image.open(file_path).convert('RGB')
        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)[0] * 100

        # Récupération des résultats
        conf, pred_idx = torch.max(probs, 0)
        pred_label = self.class_names[pred_idx.item()]
        conf_val = conf.item()

        # Mise à jour UI
        color = "#27ae60" if conf_val > 80 else "#e67e22"  # Vert si sûr, Orange sinon
        self.lbl_pred_class.config(text=pred_label, fg=color)
        self.lbl_confidence.config(text=f"Confiance : {conf_val:.1f}%")

        # Mise à jour des stats détaillées
        stats = ""
        # On trie pour afficher les plus probables en premier
        sorted_indices = torch.argsort(probs, descending=True)

        for idx in sorted_indices:
            label = self.class_names[idx]
            score = probs[idx].item()
            # Petit graphique ASCII pour le style
            bar_len = int(score / 5)
            bar = "█" * bar_len
            stats += f"{label[:15]:<15} {score:>5.1f}% {bar}\n"

        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, stats)
        self.stats_text.config(state=tk.DISABLED)


if __name__ == "__main__":
    root = tk.Tk()
    app = PlaneClassifierApp(root)
    root.mainloop()