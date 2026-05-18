"""
Few-shot image classification using a pretrained ResNet backbone
with cosine similarity matching against support set embeddings.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from tqdm import tqdm

IMAGES_DIR = Path("images")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Identity()
    model.eval().to(DEVICE)
    return model


def embed_image(model, path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = model(x)
    return feat.squeeze(0).cpu()


def embed_batch(model, paths: list[Path]) -> torch.Tensor:
    return torch.stack([embed_image(model, p) for p in tqdm(paths, leave=False)])


def cosine_classify(query_emb: torch.Tensor, support_embs: torch.Tensor, support_labels: np.ndarray) -> int:
    """Return the support label whose mean embedding is closest to the query."""
    unique_labels = np.unique(support_labels)
    class_protos = torch.stack([
        support_embs[support_labels == lbl].mean(0)
        for lbl in unique_labels
    ])
    sims = torch.nn.functional.cosine_similarity(query_emb.unsqueeze(0), class_protos)
    return int(unique_labels[sims.argmax().item()])


def main():
    episodes = pd.read_csv("test_episodes_release.csv")
    submission = pd.read_csv("sample_submission.csv")

    model = load_model()
    print(f"Running on {DEVICE}")

    predictions: dict[int, int] = {}

    for ep_id, group in tqdm(episodes.groupby("episode_id"), desc="Episodes"):
        support = group[group["role"] == "support"]
        query = group[group["role"] == "query"]

        support_paths = [IMAGES_DIR / fn for fn in support["filename"]]
        query_paths = [IMAGES_DIR / fn for fn in query["filename"]]

        support_embs = embed_batch(model, support_paths)
        query_embs = embed_batch(model, query_paths)
        support_labels = support["label"].to_numpy()

        for row_id, q_emb in zip(query.index, query_embs):
            pred = cosine_classify(q_emb, support_embs, support_labels)
            row_id_val = episodes.loc[row_id, "Id"] if "Id" in episodes.columns else row_id
            predictions[row_id_val] = pred

    submission["label"] = submission["Id"].map(predictions)
    submission.to_csv("submission.csv", index=False)
    print("Saved submission.csv")


if __name__ == "__main__":
    main()
