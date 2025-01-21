
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from tqdm import tqdm


class MLMtrainer:
    def __init__(self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            weight_save_path: str = "best_pre_train_weights.pt"):


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        print(f"Training on device {self.device} and it is awesome!!!")

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criteronMLM = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr = 5e-5) # TODO: How to (and where) to introduce a lr-scheduler?
        self.best_val_loss = float('inf')

    def _run_epoch(self, epoch_nr):

        self.model.train()
        total_loss, mlm_correct, total_mlm = 0, 0, 0
        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch_nr + 1}", leave=False)

        ignore_index = self.train_loader.dataset.ignore_index

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)  # Fixed misplaced parentheses
            mlm_labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()
            mlm_logits = self.model(input_ids, attention_mask)

            # Compute MLM loss
            loss = self.criteronMLM(
                mlm_logits.view(-1, self.model.mlm_head.out_features),
                mlm_labels.view(-1)
            )

            loss.backward()
            self.optimizer.step()

            # Accumulate total loss
            total_loss += loss.item()

            # Compute MLM accuracy
            mlm_preds = mlm_logits.argmax(dim=-1)
            mlm_correct += (mlm_preds == mlm_labels).masked_select(mlm_labels != ignore_index).sum().item()
            total_mlm += (mlm_labels != ignore_index).sum().item()

        # Calculate average loss and MLM accuracy
        avg_loss = total_loss / len(self.train_loader)
        mlm_acc = mlm_correct / total_mlm if total_mlm > 0 else 0

        return avg_loss, mlm_acc

    def _validate_epoch(self, epoch_nr):
        self.model.eval()
        total_loss, mlm_correct, total_mlm = 0, 0, 0
        progress_bar = tqdm(self.val_loader, desc=f"Validation Epoch {epoch_nr + 1}", leave=False)

        ignore_index = self.val_loader.dataset.ignore_index
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            mlm_labels = batch["labels"].to(self.device)

            with torch.no_grad():
                mlm_logits = self.model(input_ids, attention_mask)

                # Compute MLM loss
                loss = self.criteronMLM(
                    mlm_logits.view(-1, self.model.mlm_head.out_features),
                    mlm_labels.view(-1)
                )

                # Accumulate total loss
                total_loss += loss.item()

                # Compute MLM accuracy
                mlm_preds = mlm_logits.argmax(dim=-1)
                mlm_correct += (mlm_preds == mlm_labels).masked_select(mlm_labels != ignore_index).sum().item()
                total_mlm += (mlm_labels != ignore_index).sum().item()

        # Calculate average loss and MLM accuracy
        avg_loss = total_loss / len(self.val_loader)
        mlm_acc = mlm_correct / total_mlm if total_mlm > 0 else 0

        return avg_loss, mlm_acc


    def train(self, num_epochs = 10):
        for epoch_nr in range(num_epochs):
            # Run training for one epoch
            train_avg_loss, train_mlm_acc = self._run_epoch(epoch_nr)
            val_avg_loss, val_mlm_acc = self._validate_epoch(epoch_nr)

            if val_avg_loss < self.best_val_loss:
                self.best_val_loss = val_avg_loss
                torch.save(self.model.state_dict(), "best_MLM_weights.pt")

            # Report training metrics
            print(f"Epoch {epoch_nr + 1}/{num_epochs}")
            print(f"Train Avg Loss: {train_avg_loss:.4f}, Train MLM Accuracy: {train_mlm_acc:.4f}")
            print(f"Val Avg Loss: {val_avg_loss:.4f}, Val MLM Accuracy: {val_mlm_acc:.4f}")


class ClassificationTrainer:
    def __init__(self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            weight_save_path: str = "best_classification_weights.pt"):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        print(f"Training on device {self.device} and it is awesome!!!")

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=5e-6)
        self.best_val_loss = float('inf')

    def _run_epoch(self, epoch_nr):
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch_nr + 1}", leave=False)

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["encoded_label"].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)

            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += len(labels)

    def _validate_epoch(self, epoch_nr):
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        progress_bar = tqdm(self.val_loader, desc=f"Validation Epoch {epoch_nr + 1}", leave=False)

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["encoded_label"].to(self.device)

            with torch.no_grad():
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)

                total_loss += loss.item()
                correct += (logits.argmax(dim=-1) == labels).sum().item()
                total += len(labels)

        return total_loss / len(self.val_loader), correct / total

    def train(self, num_epochs=10):
        for epoch_nr in range(num_epochs):
            self._run_epoch(epoch_nr)
            val_loss, val_acc = self._validate_epoch(epoch_nr)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), "best_classification_weights.pt")

            print(f"Epoch {epoch_nr + 1}/{num_epochs}")
            print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")