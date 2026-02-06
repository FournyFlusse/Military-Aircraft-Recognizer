# Military Aircraft Recognizer

Application de bureau capable de reconnaître différents types d'avions de chasse (Rafale, Mirage, F-35, Su-35) grâce à l'IA.


## 🛠️ Technologies
* **Moteur IA :** PyTorch (ResNet18 Fine-Tuned)
* **Interface :** Tkinter (Python)
* **Précision :** ~99% sur le dataset de test.

## 📊 Dataset & Compliance (Flickr API)
Ce projet utilise l'API Flickr pour la constitution du dataset d'entraînement. Conformément aux Conditions d'utilisation des API Flickr :

* Usage Machine Learning : Ce projet utilise une Clé API Commerciale, comme requis par Flickr pour toute application liée à l'apprentissage automatique ou à l'IA, quel que soit le but lucratif ou non.

* Respect des Licences : Seules les images sous licences Creative Commons autorisant la réutilisation sont utilisées pour l'entraînement.

* Attribution : Ce produit utilise l'API Flickr mais n'est ni approuvé ni certifié par SmugMug, Inc.

* Protection des données : Le script de collecte respecte la règle de suppression sous 24h pour tout contenu retiré de la plateforme par ses auteurs.


## 🚀 Installation

1. Clonez le repo :
   ```bash
   git clone [https://github.com/FournyFlusse/Military-Aircraft-Recognizer.git](https://github.com/FournyFlusse/Military-Aircraft-Recognizer.git)

Installez les dépendances :

```bash
pip install -r requirements.txt
```

Lancez l'application :

```bash
python app_gui.py
```
## Le Modèle
Le modèle est basé sur un ResNet18 pré-entraîné sur ImageNet, puis fine-tuné sur un dataset spécifique d'avions militaires avec Data Augmentation.
