# development notes - eye-disease-screener

Started this because diabetic retinopathy is one of the most preventable causes of blindness - the bottleneck isn't treatment, it's access to screening. Wanted to build something a clinic could use without needing a specialist on site.

First version used a basic 10-feature pipeline - colour channel stats, a rough haemorrhage proxy, spot density. It worked on clean fundus images but fell apart on anything slightly underexposed or blurry. That's when the confidence threshold became the real engineering problem rather than the model itself. In screening, a missed case is worse than an unnecessary referral, so the threshold was set to err heavily toward sensitivity. That choice is documented in the code and isn't going to change without clinical input.

The original dashboard was a simple upload form. Added the live camera feed in 2026 using MediaPipe FaceMesh - 468 facial landmarks at 30fps in the browser, iris centre tracking using refined landmarks 468 and 473 for precise eye detection. Camera stops automatically after capture. The eye contours update in real time as you move.

First model used synthetic feature distributions for training. Switched to the APTOS 2019 dataset (real fundus images, 5 DR grades) which changed everything - the feature set had to be completely rebuilt around what actually differentiates DR grades visually rather than what was easy to compute.

Feature set went through two major iterations:

Version 1 - 10 features. Basic colour stats, rough haemorrhage and vessel proxies. Got about 53% on APTOS. Not good enough.

Version 2 - 52 features. Added multi-scale colour analysis, 16-bin grayscale histogram, centre region analysis, percentile values. Hit 57.4% CV accuracy. Improvement was real but modest.

Version 3 - 108 features. Extended histograms to 32 bins for grayscale, 16 for red and green, 8 for blue. Added ring brightness analysis (proxy for optic disc cup-to-disc ratio), clinically-motivated composite features: haemorrhage proxy (very dark pixels × red channel dominance), exudate proxy (very bright pixels that aren't green-dominant), texture score (edge variance × spot density). Model moved to 400 trees, lower learning rate, min_samples_leaf=4. Final CV accuracy: 60.5%.

The gap between 60% and the 85%+ you'd get from a fine-tuned CNN is real and documented in the roadmap. The hand-engineered approach was a deliberate choice for interpretability - you can explain exactly what each feature is measuring in clinical terms.

## what I'd do differently

- go straight to CNN - EfficientNet-B0 fine-tuned on APTOS would hit 85%+ without this amount of feature engineering effort
- image quality check before inference — currently accepts blurry or dark images and the low-confidence warning is the only safeguard
- would separate glaucoma screening into its own model rather than cramming it into the same DR grading pipeline

## known gaps

Webcam captures are for demonstration only. The model was trained on fundus photographs - a webcam image of a face produces meaningless results and the 75% confidence threshold warning exists specifically for this. Real deployment needs fundus imaging hardware.

---

*Published October 2025 with the basic upload interface and synthetic training data. Rebuilt in 2026 with real APTOS training, 108-feature pipeline and live camera feed with MediaPipe eye detection.*
