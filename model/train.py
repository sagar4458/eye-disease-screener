import os, json, random
import numpy as np
from PIL import Image
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib

DATA_DIR  = r"D:\main_projects\eye_disease_screener\data\raw\colored_images"
MODEL_DIR = r"D:\main_projects\eye_disease_screener\model"

IMG_SIZE = (224, 224)

def extract_features(img_path):
    try:
        img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
        arr = np.array(img, dtype=float)
        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        gray    = 0.299*r + 0.587*g + 0.114*b

        g_mean, g_std = g.mean(), g.std()
        r_mean, r_std = r.mean(), r.std()
        b_mean, b_std = b.mean(), b.std()
        gray_mean     = gray.mean()
        gray_std      = gray.std()

        dark_ratio    = (g < g_mean - 1.5*g_std).mean()
        very_dark     = (g < g_mean - 2.5*g_std).mean()
        spot_density  = (np.abs(gray - gray_mean) > 2*gray_std).mean()
        bright_ratio  = (gray > np.percentile(gray, 92)).mean()
        very_bright   = (gray > np.percentile(gray, 97)).mean()

        gx = np.abs(np.diff(gray, axis=1))
        gy = np.abs(np.diff(gray, axis=0))
        vessel_density  = (gx.mean() + gy.mean()) / 255.0
        edge_variance   = np.var(gx) / 1000.0
        high_edge_ratio = (gx > gx.mean() + 2*gx.std()).mean()

        gray_var  = np.var(gray) / 10000.0
        red_dom   = (r_mean - g_mean) / 255.0
        rg_ratio  = r_mean / (g_mean + 1e-6)
        rb_ratio  = r_mean / (b_mean + 1e-6)
        gb_ratio  = g_mean / (b_mean + 1e-6)

        h_bins  = np.histogram(gray.flatten(), bins=32, range=(0,255))[0] / gray.size
        r_bins  = np.histogram(r.flatten(),    bins=16, range=(0,255))[0] / r.size
        g_bins  = np.histogram(g.flatten(),    bins=16, range=(0,255))[0] / g.size
        b_bins  = np.histogram(b.flatten(),    bins=8,  range=(0,255))[0] / b.size

        center    = arr[56:168, 56:168, :]
        c_r       = center[:,:,0]
        c_g       = center[:,:,1]
        c_gray    = 0.299*c_r + 0.587*c_g + 0.114*center[:,:,2]
        c_bright  = (c_gray > np.percentile(c_gray, 90)).mean()
        c_dark    = (c_gray < np.percentile(c_gray, 10)).mean()
        c_rg_diff = (c_r.mean() - c_g.mean()) / 255.0
        c_contrast = c_gray.std() / 255.0
        c_vessel  = np.abs(np.diff(c_gray, axis=1)).mean() / 255.0

        ring_outer = arr[28:196, 28:196, :]
        ring_inner = arr[84:140, 84:140, :]
        ro_g = ring_outer[:,:,1].mean() / 255.0
        ri_g = ring_inner[:,:,1].mean() / 255.0
        ring_ratio = ro_g / (ri_g + 1e-6)

        p10, p25, p50, p75, p90 = [np.percentile(gray, p)/255.0 for p in [10,25,50,75,90]]
        iqr = p75 - p25

        hemo_proxy   = very_dark * red_dom
        exudate_proxy = very_bright * (1 - rg_ratio)
        texture_score = edge_variance * spot_density

        feats = [
            dark_ratio, very_dark, spot_density, bright_ratio, very_bright,
            vessel_density, edge_variance, high_edge_ratio,
            gray_var, red_dom, rg_ratio, rb_ratio, gb_ratio,
            r_mean/255, g_mean/255, b_mean/255,
            r_std/255,  g_std/255,  b_std/255,
            c_bright, c_dark, c_rg_diff, c_contrast, c_vessel,
            ring_ratio, ro_g, ri_g,
            p10, p25, p50, p75, p90, iqr,
            hemo_proxy, exudate_proxy, texture_score,
        ]
        feats.extend(h_bins.tolist())
        feats.extend(r_bins.tolist())
        feats.extend(g_bins.tolist())
        feats.extend(b_bins.tolist())

        return np.array(feats)
    except:
        return None

classes = sorted([
    d for d in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, d))
])
print(f"Classes: {classes}")

X, y = [], []
random.seed(42)

for cls in classes:
    cls_dir = os.path.join(DATA_DIR, cls)
    files   = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    samples = min(len(files), 500)
    files   = random.sample(files, samples)
    ok = 0
    for fname in files:
        feats = extract_features(os.path.join(cls_dir, fname))
        if feats is not None:
            X.append(feats)
            y.append(cls)
            ok += 1
    print(f"  {cls}: {ok}")

X     = np.array(X)
le    = LabelEncoder()
y_enc = le.fit_transform(y)

print(f"\nSamples : {len(X)}")
print(f"Features: {X.shape[1]}")

scaler = StandardScaler()
Xs     = scaler.fit_transform(X)

print("Training GradientBoosting...")

clf = GradientBoostingClassifier(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.04,
    subsample=0.8,
    min_samples_leaf=4,
    random_state=42
)

cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, Xs, y_enc, cv=cv, scoring="accuracy")
print(f"CV Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

clf.fit(Xs, y_enc)

joblib.dump(clf,    os.path.join(MODEL_DIR, "eye_model.joblib"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
joblib.dump(le,     os.path.join(MODEL_DIR, "label_encoder.joblib"))

with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
    json.dump({
        "cv_accuracy": round(scores.mean(), 4),
        "cv_std":      round(scores.std(), 4),
        "n_classes":   len(classes),
        "n_samples":   len(X),
        "feature_dim": int(X.shape[1]),
        "classes":     list(le.classes_),
        "model":       "GradientBoosting-400trees-104features",
        "dataset":     "APTOS 2019"
    }, f, indent=2)

print(f"Done. Accuracy: {scores.mean():.1%}")
print(f"Features: {X.shape[1]}")