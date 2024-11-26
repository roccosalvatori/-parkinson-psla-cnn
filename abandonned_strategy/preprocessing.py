import librosa
import numpy as np
import os
import pandas as pd

def calculate_max_pitch(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = [pitches[magnitudes[:, t].argmax(), t] for t in range(pitches.shape[1]) if pitches[magnitudes[:, t].argmax(), t] > 0]
    return np.max(pitch_values) if pitch_values else 0

def calculate_stdev_pitch(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = [pitches[magnitudes[:, t].argmax(), t] for t in range(pitches.shape[1]) if pitches[magnitudes[:, t].argmax(), t] > 0]
    return np.std(pitch_values) if pitch_values else 0

def process_folder(folder_path, label):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(folder_path, filename)
            max_pitch = calculate_max_pitch(file_path)
            stdev_pitch = calculate_stdev_pitch(file_path)
            data.append([max_pitch, stdev_pitch, label])
    return data

def save_prepared_data(folder_hc, folder_pd, output_csv):
    hc_data = process_folder(folder_hc, 0)  # Label 0 pour les patients sains
    pd_data = process_folder(folder_pd, 1)  # Label 1 pour les patients atteints de Parkinson
    all_data = hc_data + pd_data
    df = pd.DataFrame(all_data, columns=["MaxPitch", "StdevPitch", "Label"])
    df.to_csv(output_csv, index=False)
    print(f"Données sauvegardées dans {output_csv}")

# Exemples de chemins (à adapter)
folder_hc = "/home/selkhalifi/Bureau/projet_TIP/HC_AH"
folder_pd = "/home/selkhalifi/Bureau/projet_TIP/PD_AH"
output_csv = "prepared_data.csv"

save_prepared_data(folder_hc, folder_pd, output_csv)
