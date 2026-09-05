# =====================================================================
#  KAGGLE — Comparaison de backbones pour le commentaire 2.9
#  Entraîne YOLO11 + YOLOv12 (vrais modèles) avec la MÊME recette que
#  ton YOLOv8n, puis sort des VRAIS chiffres (mAP, par classe) + un CSV.
#
#  NB: "YOLO26" n'existe pas (série réelle: v8 -> v9 -> v10 -> 11 -> v12).
#      On compare donc contre YOLO11 et YOLOv12. (v10 en option ci-dessous.)
#
#  Environnement Kaggle:
#    - Settings -> Accelerator: GPU T4 x1  (PAS P100)
#    - Settings -> Persistence: "Files only" (pour garder /kaggle/working)
# =====================================================================

# ---------- 1) Installation ----------
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=True)

from ultralytics import YOLO
import pandas as pd, os, json

# ---------- 2) CONFIG — À ADAPTER ----------
# Chemin de TON data.yaml (le même que pour YOLOv8n).
# data.yaml doit pointer vers tes images/labels RAF-DB detection-cast.
DATA = "/kaggle/input/TON-DATASET/data.yaml"   # <-- CHANGE ICI

EPOCHS   = 100          # même que ton YOLOv8n
IMGSZ    = 256          # même letterbox 256x256
BATCH    = 16
OUT      = "/kaggle/working/backbone_compare"
os.makedirs(OUT, exist_ok=True)

# Recette IDENTIQUE à ton YOLOv8n (pour une comparaison honnête, "matched recipe")
TRAIN_ARGS = dict(
    data=DATA, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
    optimizer="SGD", lr0=0.01, momentum=0.937, cos_lr=True,
    amp=False, workers=2, seed=0, deterministic=True, verbose=True,
)

# Les modèles à comparer (poids pré-entraînés COCO, warm-start comme ton v8n)
MODELS = {
    "YOLOv8n  (proposed backbone)": "yolov8n.pt",   # référence (déjà la tienne)
    "YOLO11n":                      "yolo11n.pt",    # réel
    "YOLOv12n":                     "yolo12n.pt",    # réel
    # "YOLOv10n":                   "yolov10n.pt",   # optionnel: décommente si tu veux 3
}

CLASSES = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

# ---------- 3) Boucle d'entraînement + évaluation ----------
rows = []
per_class = {}

for label, weights in MODELS.items():
    print("\n" + "="*70)
    print(f"  TRAINING: {label}  ({weights})")
    print("="*70)
    name = weights.replace(".pt", "")
    try:
        model = YOLO(weights)
        model.train(project=OUT, name=name, exist_ok=True, **TRAIN_ARGS)

        # évaluation sur le split val/test défini dans data.yaml
        m = model.val(project=OUT, name=name + "_val", exist_ok=True)

        # --- Accuracy top-1 (detection-cast: 1 objet plein cadre par image) ---
        # calculée depuis la matrice de confusion d'Ultralytics.
        # cm est (nc+1)x(nc+1) : lignes = prédit, colonnes = vrai (+ background).
        acc = None
        try:
            import numpy as np
            cm = np.array(m.confusion_matrix.matrix, dtype=float)
            ncl = len(CLASSES)
            correct = np.trace(cm[:ncl, :ncl])        # bien classés (diagonale)
            true_total = cm[:, :ncl].sum()            # toutes les vraies instances
            acc = round(float(correct / true_total), 4) if true_total > 0 else None
        except Exception as e:
            print("accuracy non calculable:", e)

        row = {
            "Model":        label,
            "Accuracy":     acc,
            "mAP@0.5":      round(float(m.box.map50), 4),
            "mAP@0.5:0.95": round(float(m.box.map),   4),
            "precision":    round(float(m.box.mp),    4),
            "recall":       round(float(m.box.mr),    4),
        }
        rows.append(row)
        print(">>> RESULT:", row)

        # AP par classe (mAP@0.5 par classe) -> pour le heatmap si tu veux
        try:
            aps = [round(float(x), 4) for x in m.box.maps]  # per-class mAP@0.5:0.95
            per_class[label] = dict(zip(CLASSES, aps))
        except Exception as e:
            print("per-class non dispo:", e)

    except Exception as e:
        print(f"!!! ECHEC pour {label}: {e}")
        rows.append({"Model": label, "mAP@0.5": "FAILED",
                     "mAP@0.5:0.95": str(e)[:60], "precision": "-", "recall": "-"})

# ---------- 4) Sauvegarde des résultats ----------
df = pd.DataFrame(rows)
csv_path = os.path.join(OUT, "backbone_comparison.csv")
df.to_csv(csv_path, index=False)
with open(os.path.join(OUT, "per_class_ap.json"), "w") as f:
    json.dump(per_class, f, indent=2)

print("\n" + "="*70)
print("  TABLEAU FINAL (copie ces chiffres pour le paper)")
print("="*70)
print(df.to_string(index=False))
print("\nCSV  :", csv_path)
print("JSON :", os.path.join(OUT, "per_class_ap.json"))

# ---------- 5) Petit bar chart de comparaison (optionnel) ----------
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = df[df["mAP@0.5"] != "FAILED"]
    plt.figure(figsize=(7,4))
    plt.bar(d["Model"], d["mAP@0.5"].astype(float), color="#2b6cb0")
    plt.ylabel("mAP@0.5"); plt.title("Backbone comparison (matched recipe)")
    plt.xticks(rotation=20, ha="right"); plt.ylim(0,1); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "backbone_compare.png"), dpi=180)
    print("Figure:", os.path.join(OUT, "backbone_compare.png"))
except Exception as e:
    print("chart skip:", e)
