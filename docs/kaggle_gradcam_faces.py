# =====================================================================
#  KAGGLE — Grad-CAM / heatmaps d'attention sur VISAGES  (commentaire 2.9)
#  Produit la figure attendue par le reviewer : des visages avec une carte
#  de chaleur montrant OÙ le modèle FLEO regarde (yeux, sourcils, bouche).
#
#  Ce que tu dois fournir :
#    - CKPT   : ton modèle YOLOv8-FLEO entraîné (best.pt)
#    - IMAGES : quelques visages (idéalement 1 par émotion), n'importe quel .jpg/.png
#
#  Environnement Kaggle : GPU T4 (ou CPU, ça marche aussi pour l'inférence).
# =====================================================================
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "ultralytics", "opencv-python-headless"], check=True)

import torch, cv2, numpy as np, os, glob
from ultralytics import YOLO

# ---------- CONFIG — À ADAPTER ----------
CKPT   = "/kaggle/working/backbone_compare/yolov8-fleo/weights/best.pt"  # <-- TON modèle FLEO
IMG_DIR= "/kaggle/input/TON-DATASET/samples"   # <-- dossier avec ~7 visages (1 par émotion)
OUT    = "/kaggle/working/gradcam"
IMGSZ  = 256
ALPHA  = 0.45          # opacité du heatmap
os.makedirs(OUT, exist_ok=True)

# ---------- Charge le modèle ----------
yolo = YOLO(CKPT)
model = yolo.model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ---------- Choisit AUTOMATIQUEMENT une bonne couche cible ----------
# On capture toutes les sorties 4D (feature maps) et on garde la dernière
# avec une résolution spatiale raisonnable (>=8x8) = juste avant la tête.
feat_store = {}
def make_hook(idx):
    def _h(m, i, o):
        out = o[0] if isinstance(o, (list, tuple)) else o
        if torch.is_tensor(out) and out.dim() == 4 and out.shape[-1] >= 8:
            feat_store[idx] = out
    return _h

handles = []
try:
    layers = list(model.model)          # Ultralytics DetectionModel = nn.Sequential
except Exception:
    layers = list(model.modules())
for idx, mod in enumerate(layers):
    handles.append(mod.register_forward_hook(make_hook(idx)))

# ---------- Fonction Grad-CAM (activation-based, robuste, sans backprop) ----------
def cam_for(path):
    bgr = cv2.imread(path)
    if bgr is None:
        return None, None
    bgr = cv2.resize(bgr, (IMGSZ, IMGSZ))
    rgb = bgr[:, :, ::-1].copy()
    x = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    feat_store.clear()
    with torch.no_grad():
        _ = model(x)
    if not feat_store:
        return bgr, None
    # dernière feature map capturée = couche la plus proche de la tête
    key = max(feat_store.keys())
    f = feat_store[key][0]                       # (C,H,W)
    cam = f.pow(2).mean(0)                        # L2 sur les canaux -> (H,W)
    cam = cam.detach().cpu().numpy().astype(np.float32)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = cv2.resize(cam, (IMGSZ, IMGSZ))
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(bgr, 1 - ALPHA, heat, ALPHA, 0)
    return bgr, overlay

# ---------- Boucle sur les images ----------
paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")) +
               glob.glob(os.path.join(IMG_DIR, "*.png")) +
               glob.glob(os.path.join(IMG_DIR, "*.jpeg")))
assert paths, f"Aucune image trouvee dans {IMG_DIR}"

origs, overs, names = [], [], []
for p in paths:
    o, ov = cam_for(p)
    if ov is None:
        print("skip:", p); continue
    origs.append(o); overs.append(ov)
    names.append(os.path.splitext(os.path.basename(p))[0])
    cv2.imwrite(os.path.join(OUT, f"cam_{names[-1]}.png"), ov)

for h in handles:
    h.remove()

# ---------- Grille finale pour le paper (ligne haut = original, bas = heatmap) ----------
n = len(overs)
pad = 6
row_o = np.hstack([cv2.copyMakeBorder(im, pad, pad, pad, pad, cv2.BORDER_CONSTANT,
                    value=(255,255,255)) for im in origs])
row_h = np.hstack([cv2.copyMakeBorder(im, pad, pad, pad, pad, cv2.BORDER_CONSTANT,
                    value=(255,255,255)) for im in overs])
grid = np.vstack([row_o, row_h])
# bandeau de titres (noms d'émotion)
band = np.full((28, grid.shape[1], 3), 255, np.uint8)
cellw = row_o.shape[1] // n
for i, nm in enumerate(names):
    cv2.putText(band, nm, (i*cellw + 10, 19), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0,0,0), 1, cv2.LINE_AA)
grid = np.vstack([band, grid])
cv2.imwrite(os.path.join(OUT, "fig_gradcam_faces.png"), grid)
print("OK ->", os.path.join(OUT, "fig_gradcam_faces.png"))
print("Ligne haut = visage original ; ligne bas = heatmap d'attention FLEO.")
