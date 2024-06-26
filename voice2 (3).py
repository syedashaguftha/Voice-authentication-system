import os
import numpy as np
import scipy.io.wavfile as wav
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import speech_recognition as sr
from python_speech_features import mfcc

def record_voice(filename, duration=5):
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        print("Recording for {} seconds...".format(duration))
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source, timeout=duration, phrase_time_limit=duration)
    
    with open(filename, "wb") as f:
        f.write(audio.get_wav_data())

def extract_features(file_path):
    rate, signal = wav.read(file_path)
    mfcc_features = mfcc(signal, rate, numcep=13, nfft=2048)
    return mfcc_features

def train_model(voice_samples):
    all_features = []
    for sample in voice_samples:
        features = extract_features(sample)
        all_features.append(features)
    
    all_features = np.vstack(all_features)
    
    scaler = StandardScaler()
    all_features = scaler.fit_transform(all_features)

    gmm = GaussianMixture(n_components=16, covariance_type='diag', n_init=3)
    gmm.fit(all_features)
    
    return gmm, scaler

def verify_voice(model, scaler, voice_sample):
    features = extract_features(voice_sample)
    features = scaler.transform(features)
    
    scores = model.score(features)
    return np.mean(scores)

# Example usage
if __name__ == "__main__":
    os.makedirs("voice_samples", exist_ok=True)

    print("Register your voice by saying the phrase 'Open sesame'.")
    for i in range(3):
        record_voice(f"voice_samples/sample_{i}.wav")

    voice_samples = [f"voice_samples/sample_{i}.wav" for i in range(3)]
    model, scaler = train_model(voice_samples)
    
    print("Now, verify your voice by saying the phrase 'Open sesame'.")
    record_voice("voice_samples/test_sample.wav")
    
    score = verify_voice(model, scaler, "voice_samples/test_sample.wav")
    print(f"Verification score: {score}")
    
    threshold = -30  # Example threshold, this needs to be determined based on your data
    if score > threshold:
        print("Voice authenticated successfully.")
    else:
        print("Voice authentication failed.")
