🎵 Speech Emotion Recognition using RAVDESS
📌 Project Overview

This project implements Speech Emotion Recognition (SER) using the RAVDESS dataset
. The dataset contains audio files of actors speaking with different emotional tones such as happiness, sadness, anger, fear, surprise, calm, and neutral.

The main goal is to develop a model that can accurately classify human emotions from speech audio files, which has potential applications in:

🎙️ Human–Computer Interaction (HCI)

🤖 Virtual assistants (Alexa, Siri, Google Assistant)

🎬 Media & entertainment analytics

🧠 Mental health monitoring

📂 Dataset

Source: Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)

Structure: Audio files stored by Actor IDs

Format: .wav files

Classes: Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised

Example file naming convention:

03-01-05-01-02-02-23.wav


Where each number represents:

Modality (speech/song)

Vocal channel

Emotion

Emotional intensity

Statement

Repetition

Actor ID

⚙️ Project Workflow

Data Preprocessing

Converted .wav files into numerical features using MFCCs (Mel Frequency Cepstral Coefficients)

Normalized feature vectors for consistency

Split dataset into training and testing sets

Model Training

Tested multiple ML models:

🎯 MLP (Multi-Layer Perceptron)

🎯 CNN/RNN (optional extension if implemented)

Optimized using cross-validation and hyperparameter tuning

Evaluation

Metrics used: Accuracy, Precision, Recall, F1-score

Confusion matrix to analyze per-class performance

Deployment (optional)

Built an interactive pipeline to test real-time audio inputs

📊 Results

Achieved XX% accuracy on the RAVDESS test set

Most common misclassifications were between Calm vs Neutral and Happy vs Surprised

Demonstrated feasibility of using MFCC + ML models for emotion recognition

🛠️ Tech Stack

Language: Python

Libraries:

librosa – audio feature extraction

numpy, pandas – data handling

matplotlib, seaborn – visualization

scikit-learn, tensorflow/keras – ML/DL models

🚀 How to Run

Clone this repository:

git clone https://github.com/Shreyaparab28/Speech-Emotion-Detection.git
cd Speech-Emotion-Detection


Install dependencies:

pip install -r requirements.txt


Run preprocessing & training:

python train.py


Test with custom audio:

python predict.py path/to/audio.wav

📌 Future Improvements

Experiment with transformer-based architectures (e.g., Wav2Vec2, HuBERT)

Add real-time inference with microphone input

Expand dataset beyond RAVDESS for better generalization

Deploy as a Flask/FastAPI web app

📖 References

Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). Zenodo.

Librosa Documentation

Keras Documentation
