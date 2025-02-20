
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
            ######################################################################################################
            # Runs the forward pass under ``autocast``.
            
            # with torch.autocast(device_type=self.device, dtype=torch.float16):
            #     mlm_logits = self.model(input_ids, attention_mask)
            #     # output is float16 because linear layers ``autocast`` to float16.
            #     assert mlm_logits.dtype is torch.float16

            #     loss = self.criteronMLM(
            #         mlm_logits.view(-1, self.model.mlm_head.out_features),
            #         mlm_labels.view(-1)
            #     )
            #     # loss is float32 because ``mse_loss`` layers ``autocast`` to float32.
            #     assert loss.dtype is torch.float32

            # Exits ``autocast`` before backward().
            # Backward passes under ``autocast`` are not recommended.
            # Backward ops run in the same ``dtype`` ``autocast`` chose for corresponding forward
            #######################################################################################################
            mlm_logits = self.model(input_ids, attention_mask)

            #Compute MLM loss
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


class HierarchicalClassificationTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        taxonomic_levels,     # e.g. ["phylum", "class", "order", ...]
        lr=5e-5,
        weight_save_path="best_hierarchical_weights_.pt"
    ):
        self.model = model
        self.train_loader = train_loader     # DataLoader returns list[dict] if no collate_fn is specified
        self.val_loader = val_loader
        self.taxonomic_levels = taxonomic_levels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.weight_save_path = weight_save_path
        self.best_val_loss = float('inf')

    def _run_epoch(self, epoch_nr):
        self.model.train()
        total_loss = 0

        # For accuracy tracking at each level:
        correct_dict = {lvl: 0 for lvl in self.taxonomic_levels}
        total_dict = {lvl: 0 for lvl in self.taxonomic_levels}

        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch_nr+1}", leave=False)
        for batch in progress_bar:
            
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            self.optimizer.zero_grad()

            logits_list = self.model(input_ids, attention_mask)

            loss = 0.0 # Initilize loss for this batch

            # For each taxonomic level
            for i, lvl_name in enumerate(self.taxonomic_levels):
                # Collect that level's labels from each item in the batch
                level_labels = batch["output"][lvl_name]["encoded_label"].to(self.device)

                # Compute cross-entropy loss for that level
                lvl_logits = logits_list[i]
                lvl_loss = self.criterion(lvl_logits, level_labels)
                loss += lvl_loss

            # Backprop once on the sum of all losses
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # ----- Accuracy per level -----
            for i, lvl_name in enumerate(self.taxonomic_levels):
                lvl_logits = logits_list[i]
                level_labels = batch["output"][lvl_name]["encoded_label"].to(self.device)

                preds = lvl_logits.argmax(dim=-1)
                correct_dict[lvl_name] += (preds == level_labels).sum().item()
                total_dict[lvl_name]   += level_labels.size(0)

        avg_loss = total_loss / len(self.train_loader)
        accuracy_dict = {
            lvl: correct_dict[lvl] / total_dict[lvl]
            for lvl in self.taxonomic_levels
        }
        return avg_loss, accuracy_dict

    def _validate_epoch(self, epoch_nr):
        self.model.eval()
        total_loss = 0
        correct_dict = {lvl: 0 for lvl in self.taxonomic_levels}
        total_dict = {lvl: 0 for lvl in self.taxonomic_levels}

        progress_bar = tqdm(self.val_loader, desc=f"Validation Epoch {epoch_nr+1}", leave=False)
        with torch.no_grad():
            for batch in progress_bar:
                # batch is a list of items
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                logits_list = self.model(input_ids, attention_mask)

                loss = 0.0
                for i, lvl_name in enumerate(self.taxonomic_levels):
                    level_labels = batch["output"][lvl_name]["encoded_label"].to(self.device)

                    lvl_logits = logits_list[i]
                    loss += self.criterion(lvl_logits, level_labels)

                    preds = lvl_logits.argmax(dim=-1)
                    correct_dict[lvl_name] += (preds == level_labels).sum().item()
                    total_dict[lvl_name]   += level_labels.size(0)

                total_loss += loss.item()

        avg_val_loss = total_loss / len(self.val_loader)
        accuracy_dict = {
            lvl: correct_dict[lvl] / total_dict[lvl]
            for lvl in self.taxonomic_levels
        }
        return avg_val_loss, accuracy_dict

    def train(self, num_epochs=10):
        for epoch in range(num_epochs):
            train_loss, train_acc_dict = self._run_epoch(epoch)
            val_loss, val_acc_dict = self._validate_epoch(epoch)

            # Save model if validation loss improves
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.weight_save_path)

            # Print stats
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f" Train Loss: {train_loss:.4f}")
            for lvl in self.taxonomic_levels:
                print(f"   Train Acc {lvl}: {train_acc_dict[lvl]*100:.2f}%")

            print(f" Val Loss:   {val_loss:.4f}")
            for lvl in self.taxonomic_levels:
                print(f"   Val Acc {lvl}: {val_acc_dict[lvl]*100:.2f}%")
            print("-"*40)
        

class HierarchicalClassificationTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        taxonomic_levels,  # e.g. ["phylum", "class", "order", ...]
        lr=5e-5,
        weight_save_path="best_hierarchical_weights_.pt"
    ):
        self.model = model
        self.train_loader = train_loader  # DataLoader returns list[dict] if no collate_fn is specified
        self.val_loader = val_loader
        self.taxonomic_levels = taxonomic_levels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.weight_save_path = weight_save_path
        self.best_val_loss = float('inf')

        # Storage for metrics
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = {lvl: [] for lvl in self.taxonomic_levels}
        self.val_accuracies = {lvl: [] for lvl in self.taxonomic_levels}

    def _run_epoch(self, epoch_nr):
        self.model.train()
        total_loss = 0
        correct_dict = {lvl: 0 for lvl in self.taxonomic_levels}
        total_dict = {lvl: 0 for lvl in self.taxonomic_levels}

        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch_nr+1}", leave=False)
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            self.optimizer.zero_grad()
            logits_list = self.model(input_ids, attention_mask)

            loss = 0.0
            for i, lvl_name in enumerate(self.taxonomic_levels):
                level_labels = batch["output"][lvl_name]["encoded_label"].to(self.device)
                lvl_logits = logits_list[i]
                lvl_loss = self.criterion(lvl_logits, level_labels)
                loss += lvl_loss

                preds = lvl_logits.argmax(dim=-1)
                correct_dict[lvl_name] += (preds == level_labels).sum().item()
                total_dict[lvl_name] += level_labels.size(0)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)
        accuracy_dict = {
            lvl: correct_dict[lvl] / total_dict[lvl]
            for lvl in self.taxonomic_levels
        }

        # Save metrics for plotting
        self.train_losses.append(avg_loss)
        for lvl in self.taxonomic_levels:
            self.train_accuracies[lvl].append(accuracy_dict[lvl])

        return avg_loss, accuracy_dict

    def _validate_epoch(self, epoch_nr):
        self.model.eval()
        total_loss = 0
        correct_dict = {lvl: 0 for lvl in self.taxonomic_levels}
        total_dict = {lvl: 0 for lvl in self.taxonomic_levels}

        progress_bar = tqdm(self.val_loader, desc=f"Validation Epoch {epoch_nr+1}", leave=False)
        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                logits_list = self.model(input_ids, attention_mask)

                loss = 0.0
                for i, lvl_name in enumerate(self.taxonomic_levels):
                    level_labels = batch["output"][lvl_name]["encoded_label"].to(self.device)
                    lvl_logits = logits_list[i]
                    loss += self.criterion(lvl_logits, level_labels)

                    preds = lvl_logits.argmax(dim=-1)
                    correct_dict[lvl_name] += (preds == level_labels).sum().item()
                    total_dict[lvl_name] += level_labels.size(0)

                total_loss += loss.item()

        avg_val_loss = total_loss / len(self.val_loader)
        accuracy_dict = {
            lvl: correct_dict[lvl] / total_dict[lvl]
            for lvl in self.taxonomic_levels
        }

        # Save metrics for plotting
        self.val_losses.append(avg_val_loss)
        for lvl in self.taxonomic_levels:
            self.val_accuracies[lvl].append(accuracy_dict[lvl])

        return avg_val_loss, accuracy_dict

    def train(self, num_epochs=10):
        for epoch in range(num_epochs):
            train_loss, train_acc_dict = self._run_epoch(epoch)
            val_loss, val_acc_dict = self._validate_epoch(epoch)

            # Save model if validation loss improves
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.weight_save_path)

            # Print stats
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f" Train Loss: {train_loss:.4f}")
            for lvl in self.taxonomic_levels:
                print(f"   Train Acc {lvl}: {train_acc_dict[lvl]*100:.2f}%")

            print(f" Val Loss:   {val_loss:.4f}")
            for lvl in self.taxonomic_levels:
                print(f"   Val Acc {lvl}: {val_acc_dict[lvl]*100:.2f}%")
            print("-" * 40)

        # Return metrics after training for plotting
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accuracies": self.train_accuracies,
            "val_accuracies": self.val_accuracies
        }


import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class OverlappingKmerHierarchicalClassificationTrainer:
    def __init__(
        self,
        model,              # OverlappingKmerModularBertax (in classify mode)
        train_loader,       # DataLoader for training set
        val_loader,         # DataLoader for validation set
        taxonomic_levels,   # e.g. ["phylum", "class", "order", ...]
        lr=5e-5,
        weight_save_path="best_overlapping_kmer_weights.pt"
    ):
        """
        A trainer for hierarchical classification where each sample
        can have multiple k-mer input sequences. The `model` is expected
        to be OverlappingKmerModularBertax or similar, which in classify mode
        loops over all k-mers, aggregates their CLS outputs, and then
        returns a list of logits (one per taxonomic level).
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.taxonomic_levels = taxonomic_levels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Standard cross-entropy across levels
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.weight_save_path = weight_save_path
        self.best_val_loss = float('inf')

        # Track losses and accuracies by epoch
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = {lvl: [] for lvl in self.taxonomic_levels}
        self.val_accuracies = {lvl: [] for lvl in self.taxonomic_levels}

    def _run_epoch(self, epoch_nr):
        """
        One training epoch. Loops over train_loader, does forward & backward passes,
        accumulates classification loss across each hierarchical level, and tracks accuracy.
        """
        self.model.train()
        total_loss = 0.0

        # Keep track of how many predictions are correct at each level
        correct_dict = {lvl: 0 for lvl in self.taxonomic_levels}
        total_dict = {lvl: 0 for lvl in self.taxonomic_levels}

        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch_nr+1}", leave=False)
        for batch in progress_bar:
            # Move inputs to device
            input_ids = batch["input_ids"].to(self.device)         # shape could be [batch_size, k, seq_len] or [k, seq_len]
            attention_mask = batch["attention_mask"].to(self.device)

            self.optimizer.zero_grad()

            # The model returns a list of logits: [ lvl1_logits, lvl2_logits, ... ]
            logits_list = self.model(input_ids, attention_mask)

            # Sum losses across hierarchical levels
            loss = 0.0
            for i, lvl_name in enumerate(self.taxonomic_levels):
                # 'batch["output"][lvl_name]["encoded_label"]' is the integer label for that level
                # shape could be [batch_size] or just [] if you have a single sample
                level_labels = batch["output"][lvl_name]["encoded_label"].to(self.device)

                lvl_logits = logits_list[i]
                lvl_loss = self.criterion(lvl_logits, level_labels)
                loss += lvl_loss

                # Accuracy
                preds = lvl_logits.argmax(dim=-1)
                correct_dict[lvl_name] += (preds == level_labels).sum().item()
                total_dict[lvl_name] += level_labels.numel()  # total items for this level

            # Backprop
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        # Average loss over all batches
        avg_loss = total_loss / len(self.train_loader)

        # Compute accuracy at each level
        accuracy_dict = {
            lvl: correct_dict[lvl] / total_dict[lvl]
            for lvl in self.taxonomic_levels
        }

        # Store stats for plotting
        self.train_losses.append(avg_loss)
        for lvl in self.taxonomic_levels:
            self.train_accuracies[lvl].append(accuracy_dict[lvl])

        return avg_loss, accuracy_dict

    def _validate_epoch(self, epoch_nr):
        """
        One validation epoch. No gradient updates, just forward pass and metric calculation.
        """
        self.model.eval()
        total_loss = 0.0

        correct_dict = {lvl: 0 for lvl in self.taxonomic_levels}
        total_dict = {lvl: 0 for lvl in self.taxonomic_levels}

        progress_bar = tqdm(self.val_loader, desc=f"Validation Epoch {epoch_nr+1}", leave=False)
        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                logits_list = self.model(input_ids, attention_mask)

                loss = 0.0
                for i, lvl_name in enumerate(self.taxonomic_levels):
                    level_labels = batch["output"][lvl_name]["encoded_label"].to(self.device)
                    lvl_logits = logits_list[i]

                    lvl_loss = self.criterion(lvl_logits, level_labels)
                    loss += lvl_loss

                    preds = lvl_logits.argmax(dim=-1)
                    correct_dict[lvl_name] += (preds == level_labels).sum().item()
                    total_dict[lvl_name] += level_labels.numel()

                total_loss += loss.item()

        avg_val_loss = total_loss / len(self.val_loader)
        accuracy_dict = {
            lvl: correct_dict[lvl] / total_dict[lvl]
            for lvl in self.taxonomic_levels
        }

        # Store stats for plotting
        self.val_losses.append(avg_val_loss)
        for lvl in self.taxonomic_levels:
            self.val_accuracies[lvl].append(accuracy_dict[lvl])

        return avg_val_loss, accuracy_dict

    def train(self, num_epochs=10):
        """
        Main entry point to train the model for the specified number of epochs.
        Tracks metrics, saves best weights, prints progress.
        """
        for epoch in range(num_epochs):
            train_loss, train_acc_dict = self._run_epoch(epoch)
            val_loss, val_acc_dict = self._validate_epoch(epoch)

            # Save model if validation loss improves
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.weight_save_path)

            # Print stats
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f" Train Loss: {train_loss:.4f}")
            for lvl in self.taxonomic_levels:
                print(f"   Train Acc [{lvl}]: {train_acc_dict[lvl] * 100:.2f}%")

            print(f" Val Loss:   {val_loss:.4f}")
            for lvl in self.taxonomic_levels:
                print(f"   Val Acc [{lvl}]: {val_acc_dict[lvl] * 100:.2f}%")
            print("-" * 40)

        # Return final metrics if needed
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accuracies": self.train_accuracies,
            "val_accuracies": self.val_accuracies
        }
