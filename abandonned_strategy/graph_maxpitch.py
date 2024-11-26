import librosa
import numpy as np
import os
import matplotlib.pyplot as plt

def calculate_max_pitch(audio_path):
    # Charger le fichier audio
    y, sr = librosa.load(audio_path, sr=None)

    # Extraire les fréquences fondamentales de chaque trame
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

    # Filtrer les pitches avec une magnitude minimale pour réduire le bruit
    pitch_values = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()  # Trouver la fréquence avec la plus haute magnitude dans la trame
        pitch = pitches[index, t]
        if pitch > 0:  # Ajouter uniquement les valeurs positives
            pitch_values.append(pitch)

    # Convertir en numpy array pour calculer le pitch maximal
    pitch_values = np.array(pitch_values)

    # Calculer le pitch maximal
    max_pitch = np.max(pitch_values) if len(pitch_values) > 0 else 0

    return max_pitch

def analyze_folder_maxpitch(folder_path):
    # Parcourir tous les fichiers dans le dossier
    max_pitch_values = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):  # Vérifier si le fichier est au format WAV
            file_path = os.path.join(folder_path, filename)
            max_pitch_value = calculate_max_pitch(file_path)
            max_pitch_values.append(max_pitch_value)
            print(f"{filename}: MaxPitch = {max_pitch_value:.2f} Hz")
    return max_pitch_values

def plot_max_pitch_comparison(max_pitch_values1, max_pitch_values2, label1="HC_AH", label2="PD_AH", bin_width=100):
    # Vérifier si les listes ne sont pas vides
    if len(max_pitch_values1) == 0 or len(max_pitch_values2) == 0:
        print("Une ou plusieurs listes de valeurs sont vides. Impossible de tracer l'histogramme.")
        return

    # Convertir les listes en tableaux numpy
    max_pitch_values1 = np.array(max_pitch_values1)
    max_pitch_values2 = np.array(max_pitch_values2)

    # Définir les intervalles de l'histogramme
    bins = np.arange(min(np.min(max_pitch_values1), np.min(max_pitch_values2)),
                     max(np.max(max_pitch_values1), np.max(max_pitch_values2)) + bin_width,
                     bin_width)

    # Tracer l'histogramme pour chaque groupe
    plt.figure(figsize=(10, 6))
    plt.hist(max_pitch_values1, bins=bins, color='skyblue', alpha=0.5, label=label1, edgecolor='black')
    plt.hist(max_pitch_values2, bins=bins, color='orange', alpha=0.5, label=label2, edgecolor='black')

    plt.xlabel("Valeur de MaxPitch (Hz)")
    plt.ylabel("Nombre d'échantillons")
    plt.title("Comparaison de la distribution du MaxPitch entre deux groupes")
    plt.legend()
    plt.show()

# Exécuter l'analyse et tracer l'histogramme de comparaison
folder_path1 = '/home/selkhalifi/Bureau/projet_TIP/HC_AH'
folder_path2 = '/home/selkhalifi/Bureau/projet_TIP/PD_AH'

# Analyser chaque dossier
print("Analyse du dossier HC_AH...")
max_pitch_values1 = analyze_folder_maxpitch(folder_path1)

print("\nAnalyse du dossier PD_AH...")
max_pitch_values2 = analyze_folder_maxpitch(folder_path2)

# Tracer la comparaison
plot_max_pitch_comparison(max_pitch_values1, max_pitch_values2)
