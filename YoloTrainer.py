import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

class Trainer:
    def __init__(
        self, model, train_dataset, val_dataset, batch_size=32, lr=1e-3, weight_decay=1e-5, device="cuda"
    ):
        self.device = device
        self.model = model.to(self.device)
        self.batch_size = batch_size
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device, dtype=torch.long)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(self.train_loader)
        return avg_loss

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device, dtype=torch.long)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = val_loss / len(self.val_loader)
        accuracy = correct / total
        return avg_loss, accuracy

    def train(
        self, epochs, save_path=None
    ):
        best_val_loss = float("inf")

        for epoch in range(epochs):
            train_loss = self.train_one_epoch()
            val_loss, val_accuracy = self.validate()

            print(
                f"Epoch [{epoch + 1}/{epochs}]\t"
                f"Train Loss: {train_loss:.4f}\t"
                f"Val Loss: {val_loss:.4f}\t"
                f"Val Accuracy: {val_accuracy:.4f}"
            )

            # Save the best model
            if save_path and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
                print(f"Model saved to {save_path}")

# Example usage:
# Assuming `train_dataset` and `val_dataset` are instances of YoloDataset:
#
# model = YoloModel()
# trainer = Trainer(model, train_dataset, val_dataset, batch_size=32, lr=1e-3)
# trainer.train(epochs=10, save_path="best_model.pth")
