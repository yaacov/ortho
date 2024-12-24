import argparse
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os

# Allow inport of data and model modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from data.glyph_classes import GLYPH_CLASSES
from data.glyph_image_dataset import GlyphImageDataset
from data.transform import CustomTransform
from models.glyph_classifier import MODEL_PATH, GlyphClassifier

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
NUM_EPOCHS = 100
EARLY_STOPPING_THRESHOLD = 0.01  # Minimum loss to stop training

BATCH_SIZE = 50
LEARNING_RATE = 0.001
PATIENCE = 5  # Number of epochs to wait before stopping if no improvement
LOG_DIR = "logs/tensorboard/font_classifier"


def train_model(root_dir, resume=False):
    """
    Train the model using the dataset in the specified root directory.

    Args:
        root_dir (str): Root directory containing training images.
        resume (bool): Whether to resume training from a saved model.
    """
    # Load dataset
    transform = CustomTransform()
    train_dataset = GlyphImageDataset(root_dir=root_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize the model, loss function, and optimizer
    model = GlyphClassifier().to(device)
    if resume and os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
        print("Resuming training from saved model.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=LOG_DIR)

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Log the loss
            if batch_idx % 10 == 0:
                writer.add_scalar(
                    "Training Loss", loss.item(), epoch * len(train_loader) + batch_idx
                )

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}")
        writer.add_scalar("Average Loss per Epoch", avg_loss, epoch)

        # Early stopping
        if avg_loss < EARLY_STOPPING_THRESHOLD:
            print("Early stopping triggered.")
            break

    torch.save(model.state_dict(), MODEL_PATH)  # Save the best model
    print("Training finished.")

    # Close the TensorBoard writer
    writer.close()


def main():
    """
    Main function to handle command-line arguments and run the appropriate functionality.
    """
    parser = argparse.ArgumentParser(
        description="Train a CNN to classify images into fonts."
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from a saved model."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="data/processed/letters",
        help="Root directory containing pre-labeled training images.",
    )

    args = parser.parse_args()

    train_model(args.root_dir, resume=args.resume)


if __name__ == "__main__":
    main()
