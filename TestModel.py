import pandas as pd
import os
import torchaudio
from torch.utils.data import DataLoader, Dataset
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch.nn as nn
import torch.nn.functional as F

class Config:
    AUDIO_FOLDER_TEST = "D:/5th_Computer/test"  # Update this with the correct path
    BATCH_SIZE = 16
    SAMPLE_RATE = 16000
    CHECKPOINT_PATH = "model_checkpointD17.pth"
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TestASRDataset(Dataset):
    def __init__(self, audio_folder: str):
        self.audio_folder = audio_folder
        self.audio_files = [f for f in os.listdir(audio_folder) if f.endswith('.wav')]

    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, idx: int):
        audio_file = self.audio_files[idx]
        audio_id = os.path.splitext(audio_file)[0]
        audio_path = os.path.join(self.audio_folder, audio_file)
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.squeeze(0).numpy()
        return audio_id, waveform

def collate_fn_test(batch):
    audio_ids, waveforms = zip(*batch)
    input_values = processor(list(waveforms), return_tensors="pt", padding=True, sampling_rate=Config.SAMPLE_RATE).input_values
    return audio_ids, input_values


processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-arabic")
model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-arabic").to(Config.DEVICE)

def load_checkpoint(path: str, model: nn.Module):
    checkpoint = torch.load(path, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

def generate_submission_file(audio_folder: str, output_file: str):
    # Prepare dataset and dataloader
    test_dataset = TestASRDataset(audio_folder)
    test_dataloader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn_test)
    
    model.eval()
    predictions = []

    with torch.no_grad():
        for audio_ids, input_values in test_dataloader:
            input_values = input_values.to(Config.DEVICE)
            outputs = model(input_values).logits.cpu()
            predicted_ids = torch.argmax(outputs, dim=-1)
            predicted_transcripts = processor.batch_decode(predicted_ids)
            
            for audio_id, transcript in zip(audio_ids, predicted_transcripts):
                predictions.append((audio_id, transcript))
    
    # Save predictions to a CSV file
    submission_df = pd.DataFrame(predictions, columns=["audio_id", "text"])
    submission_df.to_csv(output_file, index=False)

# Load the trained model
load_checkpoint(Config.CHECKPOINT_PATH, model)

# Generate submission file for the test set
generate_submission_file(Config.AUDIO_FOLDER_TEST, "submission.csv")
