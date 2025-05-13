import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchcrf import CRF
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import classification_report, f1_score
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR



# Label mappings
label2id = {"O": 0, "B-Peop": 1, "I-Peop": 2, "B-Org": 3, "I-Org": 4, "B-Loc": 5, "I-Loc": 6, "B-Other": 7, "I-Other": 8}
id2label = {v: k for k, v in label2id.items()}

def tokenize_and_align_labels(tokens, labels, tokenizer, label2id, max_length=128):
    tokenized_inputs = tokenizer(
        tokens,
        is_split_into_words=True,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    word_ids = tokenized_inputs.word_ids()  # list of ID of tokens for each subtoken
    previous_word_idx = None
    aligned_labels = []
    for word_id in word_ids:
        if word_id is None:  # CLS, SEP, or PAD
            aligned_labels.append(0)  # mask will ignore it
        else:
            original_label = labels[word_id]
            if word_id != previous_word_idx:
                aligned_labels.append(label2id[original_label])
                previous_word_idx = word_id
            else:
                if original_label.startswith("B-") or original_label.startswith("I-"):
                    entity_type = original_label[2:]
                    i_label = f"I-{entity_type}"
                    aligned_labels.append(label2id[i_label])
                else:
                    aligned_labels.append(label2id["O"])
    tokenized_inputs["labels"] = torch.tensor(aligned_labels)
    return tokenized_inputs

class NERDataset(Dataset):
    def __init__(self, df, tokenizer, label2id, max_length=128):
        self.df = df
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        tokens = self.df.iloc[idx]["tokens"]
        labels = self.df.iloc[idx]["bio_tags"]
        tokenized_data = tokenize_and_align_labels(
            tokens,
            labels,
            self.tokenizer,
            self.label2id,
            self.max_length
        )
        return {
            "input_ids": tokenized_data["input_ids"].squeeze(0),
            "attention_mask": tokenized_data["attention_mask"].squeeze(0),
            "labels": tokenized_data["labels"]
        }

class BertCrf(nn.Module):
    def __init__(self, num_labels, bert_name, dropout=0.4):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_name)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = CRF(num_tags=num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        emissions = self.fc(self.dropout(sequence_output))

        if labels is not None:
            log_likelihood = self.crf(emissions, labels, mask=attention_mask.bool())
            return -log_likelihood
        else:
            mask = attention_mask.bool()
            return self.crf.decode(emissions, mask=mask)

    def save_to(self, path):
        torch.save(self.state_dict(), path, _use_new_zipfile_serialization=True)

    def load_from(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu', weights_only=False))


def compute_f1(preds, labels):
    preds_flat = [tag for seq in preds for tag in seq]
    labels_flat = [
        label for seq in labels for label in seq if label != -100  # Ignore -100
    ]

    return f1_score(labels_flat, preds_flat, average='micro')


def evaluate_model(model, val_loader):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Calculate loss
            loss = model(input_ids, attention_mask, labels)
            total_loss += loss.item()

            # Get predictions
            predictions = model(input_ids, attention_mask)

            all_preds.extend(predictions)

            # Flatten labels while ignoring -100
            active_labels = [
                [label for label, mask in zip(seq_labels, seq_mask) if mask]
                for seq_labels, seq_mask in zip(labels.cpu().numpy(), attention_mask.cpu().numpy())
            ]
            all_labels.extend(active_labels)

    avg_val_loss = total_loss / len(val_loader)
    return avg_val_loss, all_preds, all_labels

def visualize_metrics(train_losses, val_losses, val_f1_scores):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss", color="blue")
    plt.plot(epochs, val_losses, label="Val Loss", color="orange")
    plt.title("Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()

def train_model(model, train_loader, val_loader, epochs=10, patience=3):
    model.train()

    train_losses, val_losses, val_f1_scores = [], [], []

    # Early stopping
    best_val_loss = float('inf')
    counter = 0

    for epoch in range(epochs):
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            loss = model(input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        #  Validation
        val_loss, val_preds, val_labels = evaluate_model(model, val_loader)
        val_f1 = compute_f1(val_preds, val_labels)
        val_losses.append(val_loss)
        val_f1_scores.append(val_f1)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val micro-F1: {val_f1:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            model.save_to('model.pt')
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

        # updating learning rate
        scheduler.step()

    visualize_metrics(train_losses, val_losses, val_f1_scores)

def get_classification_report(test_preds, test_labels, id2label):
    true_labels = []
    pred_labels = []

    for labels_seq, preds_seq in zip(test_labels, test_preds):
        for true_tag, pred_tag in zip(labels_seq, preds_seq):
            if true_tag != -100:
                true_labels.append(id2label[true_tag])
                pred_labels.append(id2label[pred_tag])

    report = classification_report(true_labels, pred_labels, output_dict=True)
    return report

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BertCrf_model = BertCrf(num_labels=len(label2id), bert_name="bert-base-uncased").to(device)

optimizer = AdamW(BertCrf_model.parameters(), lr=5e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)


    