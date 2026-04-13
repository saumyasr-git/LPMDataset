import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from glob import glob

from sentence_transformers import SentenceTransformer
from src.lpmdataset.models.shared import load_mouse_trace_with_time

# 🔥 NEW
from src.lpmdataset.data_models import SPEAKER_RESOLUTIONS, Resolution


# =========================================================
# CONFIG
# =========================================================
TOP_K = 30
SEQ_LEN = 20
EPOCHS = 50
BATCH_SIZE = 16
EMBED_DIM = 384
MAX_OCR = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sbert = SentenceTransformer('all-MiniLM-L6-v2')


# =========================================================
# RESOLUTION HELPERS (NEW)
# =========================================================
def get_resolution_from_path(path):
    parts = path.split(os.sep)
    try:
        idx = parts.index("data_oct")
        key = f"{parts[idx+1]}/{parts[idx+2]}"
    except:
        return 1280, 720

    res = SPEAKER_RESOLUTIONS.get(key, Resolution.R720P)
    return res.width, res.height


# =========================================================
# DEBUG
# =========================================================
DEBUG = {
    "attn_entropy": [],
    "attn_peak": [],
}


# =========================================================
# HELPERS
# =========================================================
def clean_text(s):
    return str(s).lower().strip()


def embed(texts):
    emb = sbert.encode(texts, show_progress_bar=False)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-6)
    return emb


def get_asr_path(mouse_path):
    return mouse_path.replace("_trace.csv", "_spoken.csv")


def get_ocr_seg_path(ocr_path):
    rel = os.path.relpath(ocr_path, "mlpdataset/data_oct")
    return os.path.join("results", "ocr_segments_regenerated", rel.replace("_ocr.csv", "_segments.json"))


def get_save_path(mouse_path):
    parts = mouse_path.split(os.sep)
    idx = parts.index("data_oct")

    dataset = parts[idx + 1]
    topic = parts[idx + 2]
    folder = parts[idx + 3]

    slide = os.path.basename(mouse_path).replace("_trace.csv", ".csv")

    save_dir = os.path.join("results", "cross_attention_res", dataset, topic, folder)
    os.makedirs(save_dir, exist_ok=True)

    return os.path.join(save_dir, slide)


def build_pairs(roots):
    pairs = []
    for root in roots:
        traces = glob(os.path.join(root, "**", "*_trace.csv"), recursive=True)
        for t in traces:
            ocr = t.replace("_trace.csv", "_ocr.csv")
            if os.path.exists(ocr):
                pairs.append((ocr, t))
    print("Pairs:", len(pairs))
    return pairs


# =========================================================
# OCR
# =========================================================
def load_json_safe(path):
    try:
        with open(path, "r") as f:
            txt = f.read().strip()
        txt = txt.replace(",]", "]").replace(",}", "}")
        return json.loads(txt)
    except:
        return []


def get_ocr_embeddings(path):
    if not os.path.exists(path):
        return None

    segs = load_json_safe(path)
    texts = [clean_text(s["text"]) for s in segs if s.get("text")]

    if len(texts) == 0:
        return None

    return embed(texts)


# =========================================================
# OCR BOXES
# =========================================================
def load_boxes(path):
    df = pd.read_csv(path)
    df = df[df["conf"] > 0]
    return [(r["left"], r["top"], r["width"], r["height"]) for _, r in df.iterrows()][:TOP_K]


def build_regions(boxes):
    centers = []
    for (l, t, w, h) in boxes:
        centers.append([l + w / 2, t + h / 2])

    centers = np.array(centers)

    if len(centers) < TOP_K:
        centers = np.vstack([centers, np.zeros((TOP_K - len(centers), 2))])

    return centers


def assign_regions(mouse_pts, centers):
    d = ((mouse_pts[:, None, :] - centers[None, :, :]) ** 2).sum(2)
    return d.argmin(axis=1)


# =========================================================
# ASR
# =========================================================
def build_asr_segments(path, window=2.0):

    if not os.path.exists(path):
        return []

    df = pd.read_csv(path)

    if "Word" not in df.columns or "Start" not in df.columns:
        return []

    df = df[df["Word"].notna()]
    df = df[df["Start"].notna()]

    if len(df) == 0:
        return []

    words = df["Word"].astype(str).tolist()
    times = df["Start"].astype(float).tolist()

    max_time = max(times)

    segs = []
    t = 0

    while t < max_time:
        chunk = [w for w, ts in zip(words, times) if t <= ts < t + window]

        if chunk:
            segs.append({
                "text": " ".join(chunk),
                "start": t,
                "end": t + window
            })

        t += window

    return segs


# =========================================================
# DATASET
# =========================================================
class DatasetCrossAttention(Dataset):

    def __init__(self, pairs):
        self.data = []

        for ocr_path, mouse_path in pairs:

            boxes = load_boxes(ocr_path)
            if len(boxes) == 0:
                continue

            centers = build_regions(boxes)

            pts, times = load_mouse_trace_with_time(mouse_path)
            if len(pts) <= SEQ_LEN:
                continue

            times = times - times[0]

            # 🔥 RESOLUTION
            W, H = get_resolution_from_path(mouse_path)

            # 🔥 NORMALIZE
            pts = pts.copy()
            pts[:, 0] /= W
            pts[:, 1] /= H

            centers = centers.copy()
            centers[:, 0] /= W
            centers[:, 1] /= H

            regions = assign_regions(pts, centers)

            asr_segs = build_asr_segments(get_asr_path(mouse_path))
            if len(asr_segs) == 0:
                continue

            asr_emb = embed([clean_text(s["text"]) for s in asr_segs])

            ocr_emb = get_ocr_embeddings(get_ocr_seg_path(ocr_path))
            if ocr_emb is None:
                continue

            aligned_asr = []

            for t in range(len(pts)):
                time = times[t] - 1.5
                if time < 0:
                    time = 0

                for j, seg in enumerate(asr_segs):
                    if seg["start"] <= time < seg["end"]:
                        aligned_asr.append(asr_emb[j])
                        break
                else:
                    aligned_asr.append(np.zeros(EMBED_DIM))

            self.data.append((
                np.array(aligned_asr),
                ocr_emb,
                centers,
                regions,
                pts,
                times,
                mouse_path
            ))

        print("Slides:", len(self.data))


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):

        asr_seq, ocr_emb, centers, regions, pts, times, mouse_path = self.data[idx]

        i = np.random.randint(0, len(regions) - SEQ_LEN)

        asr = torch.tensor(asr_seq[i:i+SEQ_LEN], dtype=torch.float32)

        if len(ocr_emb) >= MAX_OCR:
            ocr = ocr_emb[:MAX_OCR]
        else:
            pad = np.zeros((MAX_OCR - len(ocr_emb), EMBED_DIM))
            ocr = np.vstack([ocr_emb, pad])

        ocr = torch.tensor(ocr, dtype=torch.float32)
        centers_t = torch.tensor(centers, dtype=torch.float32)

        mouse_point = pts[i + SEQ_LEN]
        dists = ((centers - mouse_point) ** 2).sum(1)
        topk = np.argsort(dists)[:3]

        y = torch.zeros(TOP_K)
        y[topk] = 1.0 / 3

        return asr, ocr, centers_t, y


# =========================================================
# MODEL
# =========================================================
class CrossAttentionModel(nn.Module):

    def __init__(self):
        super().__init__()

        d_model = 128

        self.asr_proj = nn.Linear(EMBED_DIM, d_model)
        self.ocr_proj = nn.Linear(EMBED_DIM, d_model)

        self.fc = nn.Sequential(
            nn.Linear(d_model + 2 * TOP_K, 128),
            nn.ReLU(),
            nn.Linear(128, TOP_K)
        )

    def forward(self, asr_seq, ocr_emb, centers):

        Q = self.asr_proj(asr_seq)
        K = self.ocr_proj(ocr_emb)

        Q = nn.functional.normalize(Q, dim=-1)
        K = nn.functional.normalize(K, dim=-1)

        scale = 15.0
        attn = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) * scale, dim=-1)

        attn_last = attn[:, -1, :]
        entropy = -torch.sum(attn_last * torch.log(attn_last + 1e-6), dim=1)
        peak = torch.max(attn_last, dim=1).values

        DEBUG["attn_entropy"].extend(entropy.detach().cpu().numpy())
        DEBUG["attn_peak"].extend(peak.detach().cpu().numpy())

        context = torch.matmul(attn, K)[:, -1, :]

        geom = centers.view(centers.shape[0], -1)

        x = torch.cat([context, geom], dim=1)

        return torch.log_softmax(self.fc(x), dim=1)


# =========================================================
# TRAIN
# =========================================================
def train(model, loader):

    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.KLDivLoss(reduction='batchmean')

    for e in range(EPOCHS):
        total = 0

        for asr, ocr, geom, y in loader:

            asr, ocr, geom, y = asr.to(device), ocr.to(device), geom.to(device), y.to(device)

            opt.zero_grad()
            out = model(asr, ocr, geom)
            loss = loss_fn(out, y)

            loss.backward()
            opt.step()

            total += loss.item()

        print(f"Epoch {e+1}: {total/len(loader):.4f}")


# =========================================================
# EVALUATE
# =========================================================
def evaluate(model, dataset):

    model.eval()

    with torch.no_grad():

        for idx, data in enumerate(dataset.data):

            asr_seq, ocr_emb, centers, regions, pts, times, mouse_path = data

            W, H = get_resolution_from_path(mouse_path)

            rows = []

            for i in range(len(regions) - SEQ_LEN):

                asr = torch.tensor(asr_seq[i:i+SEQ_LEN], dtype=torch.float32).unsqueeze(0).to(device)

                if len(ocr_emb) >= MAX_OCR:
                    ocr = ocr_emb[:MAX_OCR]
                else:
                    pad = np.zeros((MAX_OCR - len(ocr_emb), EMBED_DIM))
                    ocr = np.vstack([ocr_emb, pad])

                ocr = torch.tensor(ocr, dtype=torch.float32).unsqueeze(0).to(device)
                centers_t = torch.tensor(centers, dtype=torch.float32).unsqueeze(0).to(device)

                out = model(asr, ocr, centers_t)
                probs = torch.exp(out)

                top1 = probs.argmax(dim=1).item()

                time_val = times[i + SEQ_LEN]

                pred_x, pred_y = centers[top1]
                gold_x, gold_y = pts[i + SEQ_LEN]

                # 🔥 DENORMALIZE
                pred_x *= W
                pred_y *= H
                gold_x *= W
                gold_y *= H

                rows.append([time_val, pred_x, pred_y, gold_x, gold_y])

            if isinstance(mouse_path, str):
                out_path = get_save_path(mouse_path)
                df = pd.DataFrame(rows, columns=["time", "pred_x", "pred_y", "gold_x", "gold_y"])
                df.to_csv(out_path, index=False)

    print("\n✅ Evaluation complete.")


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    train_pairs = build_pairs([
        "mlpdataset/data_oct/anat-1",
        "mlpdataset/data_oct/anat-2",
        "mlpdataset/data_oct/bio-1",
        "mlpdataset/data_oct/bio-3",
        "mlpdataset/data_oct/bio-4",
        "mlpdataset/data_oct/dental",
        "mlpdataset/data_oct/psy-1",
        "mlpdataset/data_oct/psy-2"
    ])

    test_pairs = build_pairs([
        "mlpdataset/data_oct/ml-1",
        "mlpdataset/data_oct/speaking"
    ])

    train_ds = DatasetCrossAttention(train_pairs)
    test_ds = DatasetCrossAttention(test_pairs)

    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = CrossAttentionModel().to(device)

    MODEL_PATH = "ocr_semantic_asr_res_10_model.pth"

    if not os.path.exists(MODEL_PATH):
        train(model, loader)
        torch.save(model.state_dict(), MODEL_PATH)
    else:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    evaluate(model, test_ds)

    print("\n===== DEBUG METRICS =====")
    if len(DEBUG["attn_entropy"]) > 0:
        print("Attention Entropy:", np.mean(DEBUG["attn_entropy"]))
        print("Attention Peak:", np.mean(DEBUG["attn_peak"]))
