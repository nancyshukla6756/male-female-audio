import pyaudio
import numpy as np
import librosa
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Load the SVM classifier and scaler
svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
scaler = StandardScaler()
model_dir = 'models'
svm_file = 'svm_model.pkl'
scaler_file = 'scaler.pkl'
svm.load(model_dir + '/' + svm_file)
scaler = pd.read_pickle(model_dir + '/' + scaler_file)

# Define the audio recording parameters
duration = 3  # seconds
sample_rate = 16000
num_channels = 1
chunk_size = 1024

# Initialize the PyAudio object
audio = pyaudio.PyAudio()

# Open a stream to record audio from the microphone
stream = audio.open(format=pyaudio.paFloat32, channels=num_channels, rate=sample_rate, input=True, frames_per_buffer=chunk_size)

# Record audio for the specified duration
print('Recording...')
frames = []
for i in range(0, int(sample_rate / chunk_size * duration)):
    data = stream.read(chunk_size)
    frames.append(data)

# Close the stream
stream.stop_stream()
stream.close()
audio.terminate()

# Convert the recorded audio to a numpy array
audio_data = np.frombuffer(b''.join(frames), dtype=np.float32)

# Extract the MFCC features from the audio data
mfcc = librosa.feature.mfcc(audio_data, sr=sample_rate, n_mfcc=20)

# Flatten the MFCC features into a single row
mfcc_flat = mfcc.reshape(1, -1)

# Scale the MFCC features using the saved scaler
mfcc_scaled = scaler.transform(mfcc_flat)

# Predict the gender label of the audio using the saved SVM classifier
gender_label = svm.predict(mfcc_scaled)[0]

# Print the predicted gender label
if gender_label == 'male':
    print('The audio is male.')
elif gender_label == 'female':
    print('The audio is female.')
else:
    print('Error: Unknown gender label.')
