import torch
from models.glyph_classifier import MODEL_PATH, GlyphClassifier


def load_model(mpdel_path=MODEL_PATH):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GlyphClassifier().to(device)
    model.load_state_dict(
        torch.load(mpdel_path, weights_only=True, map_location=torch.device(device))
    )
    model.eval()
    return model
