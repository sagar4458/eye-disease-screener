from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np, base64, io, os, joblib
from PIL import Image

app  = Flask(__name__)
CORS(app)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")

_clf    = joblib.load(os.path.join(MODEL_DIR, "eye_model.joblib"))
_scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
_le     = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))

CONDITION_INFO = {
    "No_DR": {
        "display":  "No Diabetic Retinopathy",
        "risk":     "None",
        "color":    "#06d6a0",
        "urgency":  "No immediate action needed",
        "action":   "Continue annual eye checkups. Maintain healthy blood sugar and blood pressure.",
        "referral": "Routine"
    },
    "Mild": {
        "display":  "Mild Diabetic Retinopathy",
        "risk":     "Low-Moderate",
        "color":    "#ffd166",
        "urgency":  "3-6 months",
        "action":   "Ophthalmologist review recommended. Strict blood sugar control is critical.",
        "referral": "Non-urgent"
    },
    "Moderate": {
        "display":  "Moderate Diabetic Retinopathy",
        "risk":     "Moderate",
        "color":    "#f4a261",
        "urgency":  "1-3 months",
        "action":   "Retinal specialist referral. Consider laser treatment evaluation.",
        "referral": "Soon"
    },
    "Severe": {
        "display":  "Severe Diabetic Retinopathy",
        "risk":     "High",
        "color":    "#ef476f",
        "urgency":  "1-4 weeks",
        "action":   "Urgent retinal specialist referral. Significant risk of vision loss without treatment.",
        "referral": "Urgent"
    },
    "Proliferate_DR": {
        "display":  "Proliferative Diabetic Retinopathy",
        "risk":     "Very High",
        "color":    "#e63946",
        "urgency":  "Immediate",
        "action":   "Immediate ophthalmology referral. Anti-VEGF injections or vitrectomy may be required.",
        "referral": "Emergency"
    },
}

IMG_SIZE = (224, 224)

def extract_features(img: Image.Image) -> np.ndarray:
    img  = img.convert("RGB").resize(IMG_SIZE)
    arr  = np.array(img, dtype=float)
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

    h_bins = np.histogram(gray.flatten(), bins=32, range=(0,255))[0] / gray.size
    r_bins = np.histogram(r.flatten(),    bins=16, range=(0,255))[0] / r.size
    g_bins = np.histogram(g.flatten(),    bins=16, range=(0,255))[0] / g.size
    b_bins = np.histogram(b.flatten(),    bins=8,  range=(0,255))[0] / b.size

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
    ro_g       = ring_outer[:,:,1].mean() / 255.0
    ri_g       = ring_inner[:,:,1].mean() / 255.0
    ring_ratio = ro_g / (ri_g + 1e-6)

    p10, p25, p50, p75, p90 = [np.percentile(gray, p)/255.0 for p in [10,25,50,75,90]]
    iqr = p75 - p25

    hemo_proxy    = very_dark * red_dom
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


@app.route("/api/screen", methods=["POST"])
def screen():
    data = request.json or {}
    b64  = data.get("image", "")
    if not b64:
        return jsonify({"success": False, "error": "No image provided"}), 400

    try:
        raw     = base64.b64decode(b64.split(",")[-1])
        img     = Image.open(io.BytesIO(raw))
        feats   = extract_features(img).reshape(1, -1)
        feats_s = _scaler.transform(feats)

        proba      = _clf.predict_proba(feats_s)[0]
        top3       = np.argsort(proba)[::-1][:3]
        pred_class = _le.classes_[top3[0]]
        confidence = round(float(proba[top3[0]]) * 100, 1)
        info       = CONDITION_INFO.get(pred_class, {})

        low_confidence = confidence < 75.0
        warning = (
            "Low confidence — webcam images are not suitable for DR screening. "
            "Upload a retinal fundus photograph for a reliable result."
            if low_confidence else None
        )

        return jsonify({
            "success":        True,
            "condition":      info.get("display", pred_class),
            "class":          pred_class,
            "risk":           info.get("risk", "Unknown"),
            "color":          info.get("color", "#64748b"),
            "confidence":     confidence,
            "urgency":        info.get("urgency", "Consult a doctor"),
            "action":         info.get("action", "Consult an ophthalmologist."),
            "referral":       info.get("referral", "Routine"),
            "low_confidence": low_confidence,
            "warning":        warning,
            "top_predictions": [
                {
                    "condition":   CONDITION_INFO.get(_le.classes_[i], {}).get("display", _le.classes_[i]),
                    "probability": round(float(proba[i]) * 100, 1)
                }
                for i in top3
            ],
            "disclaimer": "Screening aid only. Always consult a qualified ophthalmologist for diagnosis."
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/model_info", methods=["GET"])
def model_info():
    return jsonify({
        "model":        "GradientBoostingClassifier",
        "n_estimators": 400,
        "classes":      list(_le.classes_),
        "feature_dim":  104,
        "dataset":      "APTOS 2019"
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5014)