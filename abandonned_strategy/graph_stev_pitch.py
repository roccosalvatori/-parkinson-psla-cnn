import librosa
import numpy as np
import os
import matplotlib.pyplot as plt

def calculate_stdev_pitch(audio_path):
    # Charger le fichier audio
    y, sr = librosa.load(audio_path, sr=None)

    # Extraire les fréquences fondamentales de chaque trame
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)

    # Filtrer les pitches avec une magnitude minimale pour réduire le bruit
    pitch_values = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()  # Trouver la fréquence avec la plus haute magnitude dans la trame
        pitch = pitches[index, t]
        if pitch > 0:  # Ajouter uniquement les valeurs positives
            pitch_values.append(pitch)

    # Convertir en numpy array pour calculer l'écart-type
    pitch_values = np.array(pitch_values)

    # Calculer l'écart-type des valeurs de pitch
    stdev_pitch = np.std(pitch_values)

    return stdev_pitch

def analyze_folder(folder_path):
    # Parcourir tous les fichiers dans le dossier
    stdev_pitch_values = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):  # Vérifier si le fichier est au format WAV
            file_path = os.path.join(folder_path, filename)
            stdev_pitch_value = calculate_stdev_pitch(file_path)
            stdev_pitch_values.append(stdev_pitch_value)
            print(f"{filename}: stdevPitch = {stdev_pitch_value:.2f}")
    return stdev_pitch_values

def plot_stdev_pitch_comparison(stdev_pitch_values1, stdev_pitch_values2, label1="HC_AH", label2="PD_AH", bin_width=25):
    # Définir les intervalles de l'histogramme
    bins = np.arange(min(min(stdev_pitch_values1), min(stdev_pitch_values2)),
                     max(max(stdev_pitch_values1), max(stdev_pitch_values2)) + bin_width,
                     bin_width)

    # Tracer l'histogramme pour chaque groupe
    plt.figure(figsize=(10, 6))
    plt.hist(stdev_pitch_values1, bins=bins, color='skyblue', alpha=0.5, label=label1, edgecolor='black')
    plt.hist(stdev_pitch_values2, bins=bins, color='orange', alpha=0.5, label=label2, edgecolor='black')

    plt.xlabel("Valeur de stdevPitch")
    plt.ylabel("Nombre d'échantillons")
    plt.title("Comparaison de la distribution du stdevPitch entre deux groupes")
    plt.legend()
    plt.show()

# Exécuter l'analyse et tracer l'histogramme de comparaison
folder_path1 = '/home/selkhalifi/Bureau/projet_TIP/HC_AH'
folder_path2 = '/home/selkhalifi/Bureau/projet_TIP/PD_AH'

stdev_pitch_values1 = analyze_folder(folder_path1)
stdev_pitch_values2 = analyze_folder(folder_path2)

plot_stdev_pitch_comparison(stdev_pitch_values1, stdev_pitch_values2)
