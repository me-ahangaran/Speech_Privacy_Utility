""" Speech files were anonymized using five different techniques: pitch shifting, time scaling, adding noise, 
changing F0, and timbre modification. The input file, 'Input.csv', contains a column named 'speech_path', 
which lists the file paths for the original WAV speech files. Each speech file was anonymized and stored 
in the 'anonymized' folder, with '_anonymized' appended to the original file name. The output file,
'Output.csv', provides the file paths of the anonymized speech files in the 'anonymized' column.


"""
import librosa
import numpy as np
import soundfile as sf
import os
import pyworld as pw
import pandas as pd

# parameters setting
pitch_shift_steps = 2 # (low, medium, high) = (2,3,4) [max value = 5]
time_scale = 1.2 #fixed for all states (1.2) [max value = 1]
timbre_factor = 1.1 # (low, medium, high) = (1.1,1.2,1.3) [max value = 1.4]
f0_factor = 1.1 # (low, medium, high) = (1.1,1.2,1.3) [max value = 1.4]
noise_level = 0.001 #fixed for all states (0.001) [max value = 0.00]

def create_directory(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def pitch_shift_voice_file(input_path, output_path):    

    # Load the audio file
    y, sr = librosa.load(input_path, sr=None)
    # Apply pitch shift
    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_shift_steps)
    
    # Step 2: Time-Scale Modification
    y_stretched = librosa.effects.time_stretch(y_shifted, rate=time_scale)  # Modify speed by 20%

    # Step 3: Adding Noise
    noise = noise_level * np.random.randn(len(y_stretched))
    y_noisy = y_stretched + noise

    # Step 4: Timbre Modification using pyworld
    # Step 5: f0 Modification using pyworld

    f0, sp, ap = pw.wav2world(y_noisy, sr)  # Decompose speech
    # Spectral envelope (sp) is a smooth curve that outlines the peaks of the speech spectrum and captures the resonance characteristics of the vocal tract. These characteristics are closely related to the timbre, or the unique quality, of the speaker's voice.
    # sp_timbre_mod = sp * timbre_factor  # Modify spectral envelope sp (timbre) using linear transformation
    sp_timbre_mod = np.power(sp, timbre_factor)  # Modify spectral envelope sp (timbre) using exponential transformation

    f0_mod = f0 * f0_factor
    y_timbre_modified = pw.synthesize(f0_mod, sp_timbre_mod, ap, sr)  # Reconstruct speech
        

    # Save the anonymized audio
    sf.write(output_path, y_timbre_modified, sr)



def process_csv(input_csv, output_csv):
    # Read the input CSV file
    df = pd.read_csv(input_csv)

    # Create new columns for altered file paths
    df['anonymized'] = ''
    
    # Process each n_steps value
    
    dir_name = 'anonymized'
    create_directory(dir_name)
    # Process each voice file
    for index, row in df.iterrows():
            original_voice_path = row['speech_path']
            voice_name = os.path.splitext(os.path.basename(original_voice_path))[0]
            altered_voice_name = f'{voice_name}_anonymized.WAV'
            altered_voice_path = os.path.join(dir_name, altered_voice_name)
            
            # Apply pitch shift and save the altered file
            pitch_shift_voice_file(original_voice_path, altered_voice_path)
            
            # Update the DataFrame with the new file path
            df.at[index, 'anonymized'] = altered_voice_path
            print("Processing ",voice_name)
    
    # Save the updated DataFrame to the output CSV file
    df.to_csv(output_csv, index=False)
    print(f"Output CSV saved to {output_csv}")

# Example usage
input_csv = 'Input.csv'
output_csv = 'Output.csv'
process_csv(input_csv, output_csv)
