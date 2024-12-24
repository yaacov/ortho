import torch
from models.glyph_classifier import MODEL_PATH, GlyphClassifier


def load_model(mpdel_path=MODEL_PATH):
    model = GlyphClassifier().to("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(mpdel_path, weights_only=True))
    model.eval()
    return model
