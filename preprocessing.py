import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torchaudio
from torchaudio.transforms import MelSpectrogram, TimeMasking, FrequencyMasking
import matplotlib.pyplot as plt
import librosa.display
import random

class AudioPreprocessor:
    def __init__(self, sample_rate=16000, n_mels=128, n_fft=2048, 
                 hop_length=160, win_length=400, training=True):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.training = training
        self.mel_spec = MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            f_min=0,
            f_max=8000,
            power=2.0  # Use power spectrogram
        )
        self.time_masking = TimeMasking(time_mask_param=30)
        self.freq_masking = FrequencyMasking(freq_mask_param=20)
        
        # Define recording type configurations
        self.recording_configs = {
            'default': {'segment_length': 5, 'expected_frames': 501},  # For first dataset
            'B': {'segment_length': 10, 'expected_frames': 1001},      # Reading text
            'D': {'segment_length': 5, 'expected_frames': 501},        # Syllable execution
            'FB': {'segment_length': 8, 'expected_frames': 801},       # Phrases and words
            'V': {'segment_length': 3, 'expected_frames': 301}         # Vowel phonation
        }

    def normalize_waveform(self, waveform):
        """Normalize waveform to [-1, 1] range"""
        if waveform.size > 0:
            return waveform / (np.max(np.abs(waveform)) + 1e-6)
        return waveform

    def convert_to_db(self, mel_spec):
        """Convert power/amplitude spectrogram to dB-scale"""
        return librosa.power_to_db(mel_spec, ref=np.max, top_db=80)

    def visualize_processing_steps(self, audio_path):
        """Visualize each step of the preprocessing pipeline"""
        # Load and resample audio
        waveform, sr = librosa.load(audio_path, sr=self.sample_rate)
        waveform = self.normalize_waveform(waveform)
        
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Waveform
        plt.subplot(3, 1, 1)
        plt.plot(waveform)
        plt.title('Normalized Waveform')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        
        # Process one segment
        segment_length = 5 * self.sample_rate
        segment = waveform[:segment_length]
        if len(segment) < segment_length:
            segment = np.pad(segment, (0, segment_length - len(segment)))
        
        # Convert to mel spectrogram
        segment_tensor = torch.FloatTensor(segment)
        mel_spec = self.mel_spec(segment_tensor)
        mel_spec_db = self.convert_to_db(mel_spec.squeeze().numpy())
        
        # Plot 2: Mel Spectrogram before SpecAugment
        plt.subplot(3, 1, 2)
        librosa.display.specshow(
            mel_spec_db,
            y_axis='mel',
            x_axis='time',
            sr=self.sample_rate,
            hop_length=self.hop_length,
            fmax=8000
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram (Before SpecAugment)')
        
        # Apply SpecAugment
        mel_spec_aug = self.time_masking(mel_spec)
        mel_spec_aug = self.freq_masking(mel_spec_aug)
        mel_spec_aug_db = self.convert_to_db(mel_spec_aug.squeeze().numpy())
        
        # Plot 3: Mel Spectrogram after SpecAugment
        plt.subplot(3, 1, 3)
        librosa.display.specshow(
            mel_spec_aug_db,
            y_axis='mel',
            x_axis='time',
            sr=self.sample_rate,
            hop_length=self.hop_length,
            fmax=8000
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram (After SpecAugment)')
        
        plt.tight_layout()
        plt.show()

    def set_training(self, mode=True):
        """Set the preprocessor's training mode."""
        self.training = mode
        return self

    def process_audio(self, audio_path, recording_type=None):
        """
        Process audio file to mel spectrograms with consistent output size
        """
        # Determine configuration based on recording type
        if recording_type is None:
            config = self.recording_configs['default']
        else:
            # Get first 1-2 characters to match config
            type_prefix = recording_type[:2] if recording_type.startswith('FB') else recording_type[0]
            config = self.recording_configs.get(type_prefix, self.recording_configs['default'])
        
        # Load and resample audio
        waveform, sr = librosa.load(audio_path, sr=self.sample_rate)
        waveform = self.normalize_waveform(waveform)
        
        # Calculate target length based on configuration
        target_length = config['segment_length'] * self.sample_rate
        
        # Pad or truncate waveform to target length
        if len(waveform) > target_length:
            waveform = waveform[:target_length]
        else:
            waveform = np.pad(waveform, (0, target_length - len(waveform)))
        
        # Convert to mel spectrogram
        segment_tensor = torch.FloatTensor(waveform)
        mel_spec = self.mel_spec(segment_tensor)
        
        # Convert to dB scale and normalize
        mel_spec_db = librosa.power_to_db(mel_spec.squeeze().numpy(), ref=np.max)
        mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
        
        # Ensure consistent time dimension
        expected_frames = config['expected_frames']
        current_frames = mel_spec_db.shape[1]
        
        if current_frames < expected_frames:
            # Pad if shorter
            pad_width = expected_frames - current_frames
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
        elif current_frames > expected_frames:
            # Truncate if longer
            mel_spec_db = mel_spec_db[:, :expected_frames]
        
        # Convert to tensor and add dimensions
        mel_spec_db = torch.FloatTensor(mel_spec_db)
        mel_spec_db = mel_spec_db.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        
        # Apply SpecAugment during training
        if self.training and np.random.random() > 0.5:
            mel_spec_db = self.time_masking(mel_spec_db)
            mel_spec_db = self.freq_masking(mel_spec_db)
        
        return mel_spec_db

    def visualize_model_input(self, mel_specs, label):
        """
        Visualize the exact input that goes into the model
        mel_specs shape: (num_segments, channels=1, mel_bins=128, time_steps)
        """
        num_segments = mel_specs.shape[0]
        
        # Create a figure with a subplot for each segment
        plt.figure(figsize=(15, 5 * num_segments))
        
        for i in range(num_segments):
            plt.subplot(num_segments, 1, i + 1)
            
            # Get the mel spectrogram for this segment
            mel_spec = mel_specs[i].squeeze().numpy()  # Remove channel dimension
            
            # Display the mel spectrogram
            librosa.display.specshow(
                mel_spec,
                y_axis='mel',
                x_axis='time',
                sr=self.sample_rate,
                hop_length=self.hop_length,
                fmax=8000
            )
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Segment {i+1} (Label: {"Sick" if label == 1 else "Healthy"})')
        
        plt.tight_layout()
        plt.show()
        
        # Print the exact tensor shapes
        print("\nModel Input Details:")
        print(f"Number of segments: {num_segments}")
        print(f"Full tensor shape: {mel_specs.shape}")
        print(f"Shape per segment: {mel_specs[0].shape}")
        print(f"Value range: [{mel_specs.min():.2f}, {mel_specs.max():.2f}]")
        print(f"Mean value: {mel_specs.mean():.2f}")
        print(f"Std value: {mel_specs.std():.2f}")

class ParkinsonsDataset(Dataset):
    def __init__(self, sick_dir, healthy_dir, sick_dir2=None, healthy_dir2=None, transform=None, training=True):
        self.transform = transform
        self.preprocessor = AudioPreprocessor(training=training)
        self.samples = []
        
        # Load first dataset samples
        self._load_simple_dataset(sick_dir, healthy_dir)
        
        # Load second dataset if provided
        if sick_dir2 and healthy_dir2:
            self._load_hierarchical_dataset(sick_dir2, healthy_dir2)
            
        # Shuffle samples
        random.shuffle(self.samples)

    def _load_simple_dataset(self, sick_dir, healthy_dir):
        """Load samples from the first dataset structure"""
        # Load sick samples (label 1)
        for filename in os.listdir(sick_dir):
            if filename.endswith('.wav'):
                self.samples.append((os.path.join(sick_dir, filename), 1))
        
        # Load healthy samples (label 0)
        for filename in os.listdir(healthy_dir):
            if filename.endswith('.wav'):
                self.samples.append((os.path.join(healthy_dir, filename), 0))

    def _load_hierarchical_dataset(self, sick_dir2, healthy_dir2):
        """Load samples from the second dataset structure with patient subfolders"""
        # Define valid recording types
        valid_prefixes = [
            'B1', 'B2', 'D1', 'D2', 'FB1', 'FB2',
            'VA1', 'VA2', 'VE1', 'VE2', 'VI1', 'VI2',
            'VO1', 'VO2', 'VU1', 'VU2'
        ]
        
        # Load sick samples (label 1)
        for patient_folder in os.listdir(sick_dir2):
            patient_path = os.path.join(sick_dir2, patient_folder)
            if os.path.isdir(patient_path):
                for filename in os.listdir(patient_path):
                    if filename.endswith('.wav'):
                        # Check if file starts with valid prefix
                        prefix = next((p for p in valid_prefixes if filename.startswith(p)), None)
                        if prefix:
                            file_path = os.path.join(patient_path, filename)
                            # Store (path, label, patient_id, recording_type)
                            self.samples.append((file_path, 1, patient_folder, prefix))
        
        # Load healthy samples (label 0)
        for patient_folder in os.listdir(healthy_dir2):
            patient_path = os.path.join(healthy_dir2, patient_folder)
            if os.path.isdir(patient_path):
                for filename in os.listdir(patient_path):
                    if filename.endswith('.wav'):
                        prefix = next((p for p in valid_prefixes if filename.startswith(p)), None)
                        if prefix:
                            file_path = os.path.join(patient_path, filename)
                            self.samples.append((file_path, 0, patient_folder, prefix))

    def print_dataset_stats(self):
        print("\nDataset Statistics:")
        print(f"Total number of samples: {len(self.samples)}")
        
        # Count by health status
        sick_count = sum(1 for sample in self.samples if sample[1] == 1)
        healthy_count = sum(1 for sample in self.samples if sample[1] == 0)
        print(f"Sick samples: {sick_count}")
        print(f"Healthy samples: {healthy_count}")
        
        # Separate statistics for each dataset
        first_dataset = []
        second_dataset = []
        
        for sample in self.samples:
            # If sample has more than 2 elements, it's from the second dataset
            if len(sample) > 2:
                second_dataset.append(sample)
            else:
                first_dataset.append(sample)
        
        print("\nFirst Dataset:")
        print(f"Total samples: {len(first_dataset)}")
        print(f"Sick: {sum(1 for sample in first_dataset if sample[1] == 1)}")
        print(f"Healthy: {sum(1 for sample in first_dataset if sample[1] == 0)}")
        
        if second_dataset:
            print("\nSecond Dataset:")
            print(f"Total samples: {len(second_dataset)}")
            print(f"Sick: {sum(1 for sample in second_dataset if sample[1] == 1)}")
            print(f"Healthy: {sum(1 for sample in second_dataset if sample[1] == 0)}")
            
            # Count samples by recording type
            recording_types = {}
            for sample in second_dataset:
                rec_type = sample[3]  # Recording type is at index 3
                recording_types[rec_type] = recording_types.get(rec_type, 0) + 1
            
            print("\nSamples by recording type:")
            for rec_type, count in sorted(recording_types.items()):
                print(f"{rec_type}: {count}")

    def set_training(self, mode=True):
        """Set the dataset's training mode."""
        self.preprocessor.set_training(mode)
        return self

    def visualize_first_sample(self):
        """Visualize preprocessing steps for the first sample"""
        if len(self.samples) > 0:
            first_audio_path, label = self.samples[0]
            print(f"\nVisualizing preprocessing steps for first sample (label: {'Sick' if label == 1 else 'Healthy'})")
            self.preprocessor.visualize_processing_steps(first_audio_path)

    def visualize_sample(self, idx):
        """Visualize a specific sample that would be fed to the model"""
        mel_specs, label = self.__getitem__(idx)
        print(f"\nVisualizing sample {idx}")
        self.preprocessor.visualize_model_input(mel_specs, label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Get sample data
        sample = self.samples[idx]
        audio_path = sample[0]
        label = sample[1]
        
        # Get recording type if available (from second dataset)
        recording_type = sample[3] if len(sample) > 3 else None
        
        # Process audio with recording type information
        mel_specs = self.preprocessor.process_audio(audio_path, recording_type)
        
        if mel_specs is None:
            # Handle error case
            return None
        
        if self.transform:
            mel_specs = self.transform(mel_specs)
        
        return mel_specs, torch.tensor(label, dtype=torch.long)

def custom_collate(batch):
    """Custom collate function to handle variable-sized spectrograms"""
    # Filter out None values if any
    batch = [item for item in batch if item is not None]
    
    # Separate spectrograms and labels
    specs, labels = zip(*batch)
    
    # Get the maximum time dimension
    max_length = max(spec.size(-1) for spec in specs)
    
    # Pad all spectrograms to the maximum length
    padded_specs = []
    for spec in specs:
        if spec.size(-1) < max_length:
            padding = torch.zeros(1, 1, spec.size(2), max_length - spec.size(-1))
            spec = torch.cat([spec, padding], dim=-1)
        padded_specs.append(spec)
    
    # Stack all spectrograms and labels
    specs_tensor = torch.cat(padded_specs, dim=0)
    labels_tensor = torch.tensor(labels)
    
    return specs_tensor, labels_tensor

def main():
    """Test the preprocessing pipeline and show data structure"""
    dataset = ParkinsonsDataset(
        sick_dir='./sick/',
        healthy_dir='./healthy/',
        visualize=False
    )
    
    # Get first sample
    mel_specs, label = dataset[0]
    
    print("\nSingle Sample Details:")
    print(f"Label: {'Sick' if label == 1 else 'Healthy'}")
    print(f"Mel Spectrogram Shape: {mel_specs.shape}")
    print(f"  - Number of segments: {mel_specs.shape[0]}")
    print(f"  - Channels: {mel_specs.shape[1]}")
    print(f"  - Frequency bins: {mel_specs.shape[2]}")
    print(f"  - Time steps: {mel_specs.shape[3]}")
    print(f"\nValue Range: [{mel_specs.min():.2f}, {mel_specs.max():.2f}]")
    print(f"Mean Value: {mel_specs.mean():.2f}")
    print(f"Std Value: {mel_specs.std():.2f}")
    
    # Create a batch
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate)
    batch, labels = next(iter(dataloader))
    
    print("\nBatch Details:")
    print(f"Batch Shape: {batch.shape}")
    print(f"Labels: {labels}")
    
    # Show sample values from first segment of first sample
    print("\nSample Values (first segment, first sample):")
    first_segment = batch[0][0]
    print(f"First few values:\n{first_segment[:3, :3]}")

if __name__ == '__main__':
    main()
