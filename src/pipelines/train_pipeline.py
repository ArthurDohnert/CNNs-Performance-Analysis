# src/pipelines/train_pipeline.py

###
### Functions that manage the training workflow
###

# imports
import torch
import torch.nn as nn
import torch.optim as optim
import time

# hyperparams
target_accuracy_percentage = 90
learning_rate = 0.001
num_epochs = 300


# implementations
class Trainer:
    def __init__(self, model, trainloader, testloader, device=None, lr=learning_rate, epochs=num_epochs):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.trainloader = trainloader
        self.testloader = testloader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.history = {"loss": [], "acc": []}

    # trains the model until it reaches 90% or epochs.
    def train(self, target_acc=target_accuracy_percentage):
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            start = time.time()

            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            # evaluating on test set
            acc = self.evaluate()
            elapsed = time.time() - start

            avg_loss = running_loss / len(self.trainloader)
            self.history["loss"].append(avg_loss)
            self.history["acc"].append(acc)

            print(f"Epoch {epoch+1}/{self.epochs} | Loss {avg_loss:.4f} | Test Acc {acc:.2f}% | {elapsed:.2f}s")

            # test stopping
            if acc >= target_acc:
                print(f"{acc:.2f}% accuracy on test (>= {target_acc}%)\n stopping the training...")
                break

    # evaluates the model on test set and returns the accuracy
    def evaluate(self):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in self.testloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        return 100 * correct / total