import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import soundfile as sf
from tqdm import tqdm

class RPWAudioProcessor:
    def __init__(self, sr=44100):
        """Initialize the RPW audio processor with given sample rate"""
        self.sr = sr

    def load_audio(self, file_path):
        """Load audio file with specified sample rate"""
        audio, _ = librosa.load(file_path, sr=self.sr)
        return audio

    def butter_bandpass(self, lowcut, highcut, order=5):
        """Create butterworth bandpass filter"""
        nyquist = 0.5 * self.sr
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def apply_bandpass_filter(self, data, lowcut, highcut, order=5):
        """Apply bandpass filter to audio data"""
        b, a = self.butter_bandpass(lowcut, highcut, order=order)
        filtered_data = filtfilt(b, a, data)
        return filtered_data

    def remove_background_noise(self, audio, noise_reduction_threshold=2):
        """Remove background noise using spectral gating"""
        S = librosa.stft(audio)
        mag = np.abs(S)
        
        noise_profile = np.mean(mag, axis=1) + np.std(mag, axis=1)
        mask = mag > (noise_profile[:, np.newaxis] * noise_reduction_threshold)
        S_cleaned = S * mask
        
        audio_cleaned = librosa.istft(S_cleaned)
        return audio_cleaned

    def create_spectrogram(self, audio, output_path=None):
        """Generate and save spectrogram without any decorative elements"""
        # Create figure with black background
        plt.figure(figsize=(12, 8), facecolor='blue')
        ax = plt.gca()
        ax.set_facecolor('blue')
        
        # Compute STFT
        D = librosa.stft(audio)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        # Plot spectrogram with minimal styling
        librosa.display.specshow(
            S_db,
            sr=self.sr,
            vmin=-80,
            vmax=0,
            cmap='magma'
        )
        
        # Remove all axes, labels, and decorations
        plt.axis('off')
        
        # Remove padding and margins
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0,0)
        
        if output_path:
            plt.savefig(output_path, 
                       bbox_inches='tight',
                       pad_inches=0,
                       facecolor='blue',
                       edgecolor='none',
                       dpi=300)
            plt.close()
        else:
            plt.show()
        
        return S_db

    def process_audio(self, input_path, output_audio_path=None, output_spec_path=None):
        """Process single audio file"""
        # Load and process audio
        audio = self.load_audio(input_path)
        
        # Apply filters
        filtered_audio = self.apply_bandpass_filter(audio, 100, 3000)
        cleaned_audio = self.remove_background_noise(filtered_audio)
        cleaned_audio = librosa.util.normalize(cleaned_audio)
        
        # Save cleaned audio
        if output_audio_path:
            sf.write(output_audio_path, cleaned_audio, self.sr)
        
        # Create spectrogram
        spectrogram = self.create_spectrogram(cleaned_audio, output_spec_path)
        
        return cleaned_audio, spectrogram

    def analyze_events(self, spectrogram, threshold_db=-20):
        """Analyze RPW events in the spectrogram"""
        events = spectrogram > threshold_db
        frame_times = librosa.frames_to_time(
            np.arange(events.shape[1]), 
            sr=self.sr, 
            hop_length=512
        )
        
        # Find continuous events
        event_regions = []
        current_event = None
        
        for i, frame in enumerate(events.T):
            if np.any(frame):
                if current_event is None:
                    current_event = {'start': frame_times[i], 'end': frame_times[i]}
                else:
                    current_event['end'] = frame_times[i]
            elif current_event is not None:
                event_regions.append(current_event)
                current_event = None
        
        # Calculate statistics
        event_durations = [event['end'] - event['start'] for event in event_regions]
        stats = {
            'num_events': len(event_regions),
            'total_duration': sum(event_durations),
            'mean_duration': np.mean(event_durations) if event_durations else 0,
            'events': event_regions
        }
        
        return stats

def process_directory(input_dir, output_base_dir):
    """Process all audio files in a directory"""
    # Create output directories if they don't exist
    spectrograms_dir = os.path.join(output_base_dir, 'spectrograms')
    
    os.makedirs(spectrograms_dir, exist_ok=True)
    
    # Initialize processor
    processor = RPWAudioProcessor()
    
    # Get all audio files
    audio_files = [f for f in os.listdir(input_dir) if f.endswith(('.wav', '.mp3'))]
    
    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        try:
            input_path = os.path.join(input_dir, audio_file)
            base_name = os.path.splitext(audio_file)[0]
            
            output_spec_path = os.path.join(spectrograms_dir, f"spec_{base_name}.png")
            
            # Process audio and create spectrogram
            audio = processor.load_audio(input_path)
            filtered_audio = processor.apply_bandpass_filter(audio, 100, 3000)
            cleaned_audio = processor.remove_background_noise(filtered_audio)
            cleaned_audio = librosa.util.normalize(cleaned_audio)
            processor.create_spectrogram(cleaned_audio, output_spec_path)
            
        except Exception as e:
            print(f"Error processing {audio_file}: {str(e)}")

if __name__ == "__main__":
    # Example usage
    input_dir = r"C:\Users\Tushar\OneDrive\Desktop\AgriScience\Audio_Files\Lab\Clean"
    output_dir = r"C:\Users\Tushar\OneDrive\Desktop\AgriScience\Spectogram\Lab\Clean"
    
    results = process_directory(input_dir, output_dir)
    
    # Print results
    # for result in results:
    #     print(f"\nFile: {result['file']}")
    #     print(f"Number of RPW events: {result['num_events']}")
    #     print(f"Total duration: {result['total_duration']:.2f} seconds")
    #     print(f"Average event duration: {result['mean_duration']:.2f} seconds")


 