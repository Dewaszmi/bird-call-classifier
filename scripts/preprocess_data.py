#!/usr/bin/env python3

import glob
import os

import librosa
import numpy as np


def pad_audio(y, target_length):
    length = len(y)
    if length < target_length:
        padding = target_length - length
        y = np.pad(y, (0, padding), mode='constant')
    elif length > target_length:
        y = y[:target_length]
    return y

def compute_mel_spectrogram(y, sr, n_mels=128, n_fft=2048, hop_length=512):
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return log_mel_spectrogram

def main():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'processed_data')
    
    os.makedirs(output_dir, exist_ok=True)
    
    target_sr = 22050
    target_duration = 30.0
    target_samples = int(target_sr * target_duration)
    
    species_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    for species in species_dirs:
        species_dir = os.path.join(data_dir, species)
        output_species_dir = os.path.join(output_dir, species)
        os.makedirs(output_species_dir, exist_ok=True)
        
        audio_files = []
        for ext in ('*.mp3', '*.wav'):
            audio_files.extend(glob.glob(os.path.join(species_dir, ext)))
        
        for file_path in audio_files:
            file_name = os.path.basename(file_path)
            output_file_name = os.path.splitext(file_name)[0] + '.npy'
            output_file_path = os.path.join(output_species_dir, output_file_name)
            
            if os.path.exists(output_file_path):
                continue
                
            try:
                y, sr = librosa.load(file_path, sr=target_sr, duration=target_duration)
                y = pad_audio(y, target_samples)
                
                mel_spec = compute_mel_spectrogram(y, sr)
                
                np.save(output_file_path, mel_spec)
                print(f"Processed: {species} / {file_name}")
            except Exception as e:
                print(f"Failed to process {file_name}: {e}")

if __name__ == "__main__":
    main()
