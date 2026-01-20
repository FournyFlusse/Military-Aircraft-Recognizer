# Military Aircraft Recognizer

Application de bureau capable de reconnaître différents types d'avions de chasse (Rafale, Mirage, F-35, Su-35) grâce à l'IA.


## 🛠️ Technologies
* **Moteur IA :** PyTorch (ResNet18 Fine-Tuned)
* **Interface :** Tkinter (Python)
* **Précision :** ~99% sur le dataset de test.


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
