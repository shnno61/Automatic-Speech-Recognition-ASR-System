import time
import pandas as pd
import torchaudio
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import logging
import os
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from collections import namedtuple
from typing import List, Tuple
import Levenshtein

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    TRAIN_CSV = "D:/5th_Computer/MTC_ASR_Dataset_16K/train.csv"
    ADAPT_CSV = "D:/5th_Computer/MTC_ASR_Dataset_16K/adapt.csv"
    AUDIO_FOLDER_TRAIN = "D:/5th_Computer/MTC_ASR_Dataset_16K/train"
    AUDIO_FOLDER_ADAPT = "D:/5th_Computer/MTC_ASR_Dataset_16K/adapt"
    SAMPLE_RATE = 16000
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.0001
    CHECKPOINT_PATH = "model_checkpointD17K.pth"
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SparseTensor = namedtuple('SparseTensor', 'indices vals shape')


processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-arabic")
model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-arabic").to(Config.DEVICE)

class CustomASRDataset(Dataset):
    def __init__(self, csv_file: str, audio_folder: str):
        self.data = pd.read_csv(csv_file)
        self.audio_folder = audio_folder
        self.data.dropna(subset=[self.data.columns[1]], inplace=True)
        self.data[self.data.columns[1]] = self.data[self.data.columns[1]].astype(str)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, str]:
        audio_path = os.path.join(self.audio_folder, f"{self.data.iloc[idx, 0]}.wav")
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.squeeze(0).numpy()
        transcript = self.data.iloc[idx, 1]
        return waveform, transcript

def collate_fn(batch: List[Tuple[np.ndarray, str]]) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    waveforms, transcripts = zip(*batch)
    input_values = processor(list(waveforms), return_tensors="pt", padding=True, sampling_rate=Config.SAMPLE_RATE).input_values
    transcript_lengths = [len(transcript) for transcript in transcripts]
    transcripts_encoded = [torch.tensor(processor.tokenizer(transcript).input_ids, dtype=torch.long) for transcript in transcripts]
    transcripts_padded = pad_sequence(transcripts_encoded, batch_first=True, padding_value=processor.tokenizer.pad_token_id)
    return input_values, transcripts_padded, transcript_lengths

def calc_PER(pred, ground_truth, normalize=True):
    pred_seq_list = ["".join(pred)]
    truth_seq_list = ["".join(ground_truth)]

    assert len(truth_seq_list) == len(pred_seq_list)

    distances = []
    for i in range(len(truth_seq_list)):
        dist_i = Levenshtein.distance(pred_seq_list[i], truth_seq_list[i])
        if normalize:
            dist_i /= float(len(truth_seq_list[i]))
        distances.append(dist_i)

    return np.mean(distances)

def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int, loss: float, path: str):
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
    }
    torch.save(state, path)
    logger.info(f"Checkpoint saved at epoch {epoch}, loss: {loss}")

def load_checkpoint(path: str, model: nn.Module, optimizer: optim.Optimizer) -> Tuple[int, float]:
    if not torch.cuda.is_available():
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    logger.info(f"Checkpoint loaded from epoch {epoch}, loss: {loss}")
    return epoch, loss

def train_model(model, processor, train_dataloader, num_epochs, optimizer, start_epoch=0):
    start_time = time.time()
    model.to(Config.DEVICE)
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0
        for batch_idx, (input_values, transcript, transcript_lengths) in enumerate(train_dataloader):
            optimizer.zero_grad()
            input_values = input_values.to(Config.DEVICE)
            transcript = transcript.to(Config.DEVICE)

            outputs = model(input_values, labels=transcript)
            loss = outputs.loss

            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}/{len(train_dataloader)}, Loss: {loss.item()}")
        
        avg_loss = epoch_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, optimizer, epoch + 1, avg_loss, Config.CHECKPOINT_PATH)

            # Calculate PER on a subset of the training data
            model.eval()
            with torch.no_grad():
                all_preds = []
                all_truths = []
                for batch_idx, (input_values, transcript, transcript_lengths) in enumerate(train_dataloader):
                    if batch_idx >= 5:  # Use only a subset of the data for PER calculation
                        break
                    input_values = input_values.to(Config.DEVICE)
                    outputs = model(input_values).logits.cpu()
                    predicted_ids = torch.argmax(outputs, dim=-1)
                    predicted_transcripts = processor.batch_decode(predicted_ids)

                    for i in range(len(transcript)):
                        pred = predicted_transcripts[i]
                        truth = processor.batch_decode(transcript[i].unsqueeze(0))
                        all_preds.append(pred)
                        all_truths.append(truth[0])
                
                if all_preds and all_truths:
                    per = calc_PER(all_preds, all_truths)
                    logger.info(f"Phoneme Error Rate (PER) after epoch {epoch + 1}: {per}")
    end_time = time.time()
    logger.info(f"Training Time : {end_time - start_time} seconds")

    # Print reference and predicted sentences
    model.eval()
    with torch.no_grad():
        for batch_idx, (input_values, transcript, transcript_lengths) in enumerate(train_dataloader):
            if batch_idx >= 5:  # Limit the number of batches for demonstration
                break
            input_values = input_values.to(Config.DEVICE)
            outputs = model(input_values).logits.cpu()
            predicted_ids = torch.argmax(outputs, dim=-1)
            predicted_transcripts = processor.batch_decode(predicted_ids)

            for i in range(len(transcript)):
                reference = processor.batch_decode(transcript[i].unsqueeze(0))
                prediction = predicted_transcripts[i]
                print("-" * 100)
                print("Reference:", reference[0])
                print("Prediction:", prediction)

if __name__ == "__main__":
    # Ensure deterministic behavior for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Prepare datasets
    train_dataset = CustomASRDataset(Config.TRAIN_CSV, Config.AUDIO_FOLDER_TRAIN)
    adapt_dataset = CustomASRDataset(Config.ADAPT_CSV, Config.AUDIO_FOLDER_ADAPT)

    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    adapt_dataloader = DataLoader(adapt_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    logger.info("Data loaders created.")

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    logger.info("Optimizer defined.")

    # Load checkpoint if exists
    start_epoch = 0
    if os.path.exists(Config.CHECKPOINT_PATH):
        start_epoch, _ = load_checkpoint(Config.CHECKPOINT_PATH, model, optimizer)

    # Train model
    train_model(model, processor, train_dataloader, Config.NUM_EPOCHS, optimizer, start_epoch)
    logger.info("Training complete.")
