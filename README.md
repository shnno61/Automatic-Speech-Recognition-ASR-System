# Project Overview

- **Project Title:** Automatic Speech Recognition (ASR) System
- **Project Team:**
  1. Samer Ahmed Eissa
  2. Hamdy Waleed Hamdy Adelhaleem
  3. Ahmed Adel Adbelallah Atia Eissa
  4. Ahmed Ibrahim ELkholy
  5. Omar Khaled Abdelhaleem Omran

## Objective

The objective of this project is to develop an advanced ASR system capable of accurately transcribing speech into text in real-time for Egyptian Arabic language, leveraging state-of-the-art models for feature extraction, acoustic modeling, and language modeling.

For any ASR system it should contain 3 main parts (models):
1. **Feature Extraction Model**
   - **Model:** Mel Frequency Cepstral Coefficients (MFCC)
   - **Reasons for Use:**
     - **Robustness:** MFCCs are robust in capturing the spectral characteristics of speech, making them suitable for a wide range of acoustic environments.
     - **Computational Efficiency:** They are computationally efficient, allowing for real-time processing of audio signals.
     - **Generalization:** Widely adopted in ASR systems, ensuring compatibility and comparability across different research and applications.

2. **Acoustic Model**
   - **Model:** CNN-Conformer
   - **Reasons for Use:**
     - **Hierarchical Feature Learning:** The combination of CNNs and the Transformer-based Conformer architecture enables effective hierarchical feature learning from raw audio signals, improving robustness and accuracy in acoustic modelling.
     - **Contextual Adaptation:** Conformers integrate self-attention mechanisms, allowing the model to dynamically adapt to varying contexts within speech signals, enhancing transcription accuracy.
     - **Long term memory:** Due to Arabic grammar, the next state in the sentence does not depend on the current state, so we chose this model as it supports long-term memory feature.
     - **Performance:** CNN-Conformer models have demonstrated superior performance compared to traditional acoustic models, especially in handling long-range dependencies and complex acoustic patterns.

3. **Language Model**
   - **Model:** Wave2Vec (jonatasgrosman/wav2vec2-large-xlsr-53-arabic)
   - **Reasons for Use:**
     - **Contextual Speech Representation:** Wave2Vec learns contextually rich representations directly from waveform data, capturing fine-grained acoustic and linguistic features crucial for accurate transcription.
     - **Self-Supervised Learning:** Trained in a self-supervised manner, Wave2Vec leverages large unlabeled audio corpora, reducing the need for labeled data and improving generalization.
     - **Pre-trained for Arabic:** Earlier, Facebook trained this model to be capable for Arabic language, so we chose it to deal with Egyptian Arabic language.
     - **State-of-the-Art Performance:** Wave2Vec has achieved state-of-the-art results in language modeling tasks, surpassing traditional approaches by effectively leveraging deep learning techniques tailored for speech recognition.

## Implementation Details

- **Data Preprocessing:** Clearing noise from dataset used for the model is crucial for ensuring accurate transcription of speech signals. Noise can distort speech signals, making it challenging for ASR systems to accurately recognize and transcribe spoken words. We used simple noise reduction algorithm spectral subtraction with fine tuning that works as noise cancellation.

- **Model Training:** Sets up data loading from CSV files containing audio paths and transcripts, processes audio waveforms using Torchaudio, and tokenizes transcripts using a pre-trained Wav2Vec2 processor. The model, pre-trained on Arabic data (jonatasgrosman/wav2vec2-large-xlsr-53-arabic), is fine-tuned using an Adam optimizer with configurable batch size and learning rate. Training progresses over specified epochs, logging loss metrics and saving checkpoints (`model_checkpointD17.pth`) every 5 epochs. Checkpoints enable resuming training and evaluating model performance using Phoneme Error Rate (PER).

## Challenges Faced

- **Data Variability:** The Egyptian Arabic language presents challenges due to its complexity in vocabulary and grammar diversity within the dataset. Therefore, our approach focuses on training a model capable of effectively handling this variability and generalizing across diverse linguistic data.

- **Model Complexity:** Achieving simplicity and accessibility is our goal, so we aimed to create a straightforward ASR model with reduced complexity. Simplifying an ASR model is a challenging task, but we successfully achieved this goal with our approach.

- **Resource Constraints:** Limitation of dataset and the necessity to build the acoustic model from scratch (unable to use pre-trained models) were our hard challenges, but we handled them and moved forward.
