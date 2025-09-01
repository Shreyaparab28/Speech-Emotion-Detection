# 🎵 Speech Emotion Recognition using RAVDESS

## 📌 Project Overview  
This project implements **Speech Emotion Recognition (SER)** using the [RAVDESS dataset](https://zenodo.org/record/1188976).  
The aim is to classify human emotions from speech audio files.  

### 🔥 Applications:
- 🎙️ Human–Computer Interaction (HCI)  
- 🤖 Virtual assistants (Alexa, Siri, Google Assistant)  
- 🎬 Media & entertainment analytics  
- 🧠 Mental health monitoring  

---

## 📂 Dataset  
- **Source:** Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)  
- **Format:** `.wav` audio files organized by actor folders  
- **Emotions Covered:** Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised  

**Example filename:**  
03-01-05-01-02-02-23.wav

Where each number represents:  
1. Modality (speech/song)  
2. Vocal channel  
3. Emotion  
4. Emotional intensity  
5. Statement  
6. Repetition  
7. Actor ID  

---

## ⚙️ Project Workflow  
1. **Data Preprocessing**  
   - Converted `.wav` files into numerical features using **MFCCs (Mel Frequency Cepstral Coefficients)**  
   - Normalized feature vectors  
   - Split dataset into train/test sets  

2. **Model Training**  
   - Implemented models such as:  
     - 🎯 Multi-Layer Perceptron (MLP)  
     - 🎯 CNN/RNN (optional extension)  
   - Used cross-validation and hyperparameter tuning  

3. **Evaluation**  
   - Metrics: **Accuracy, Precision, Recall, F1-score**  
   - Confusion matrix to analyze per-class performance  

---

## 📊 Results  
- Achieved **XX% accuracy** on test set  
- Common misclassifications: **Calm vs Neutral**, **Happy vs Surprised**  
- Demonstrated feasibility of using MFCC + ML/DL models for speech emotion detection  

---

## 🛠️ Tech Stack  
- **Language:** Python  
- **Libraries:**  
  - `librosa` – audio feature extraction  
  - `numpy`, `pandas` – data handling  
  - `matplotlib`, `seaborn` – visualization  
  - `scikit-learn`, `tensorflow/keras` – ML/DL models  

---

## 🚀 How to Run  
1. Clone this repository:  
   ```bash
   git clone https://github.com/Shreyaparab28/Speech-Emotion-Detection.git
   cd Speech-Emotion-Detection

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the model:
```bash
python train.py
```

4. Test with custom audio:
```bash
python predict.py path/to/audio.wav
```

📖 References

- Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). Zenodo.

- Librosa Documentation

- Keras Documentation

