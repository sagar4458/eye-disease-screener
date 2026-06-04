# roadmap - eye-disease-screener

## where it is now

108-feature GradientBoosting pipeline trained on APTOS 2019. CV accuracy 60.5% across 5 DR grades. Live camera feed with MediaPipe iris tracking. Confidence threshold at 75% with explicit warning for low-quality inputs. Runs fully local - no cloud dependency.

## the obvious next step - CNN

The hand-engineered feature approach has hit its ceiling. The gap between 60% and what's clinically useful (85%+) won't be closed with more features - it needs a CNN that learns what to look at directly from the images.

Plan is EfficientNet-B0 fine-tuned on the full APTOS dataset. Small enough to run on a laptop GPU, well-documented fine-tuning path and there are public benchmarks to compare against. Expected accuracy: 87-92% on APTOS test set.

The feature engineering work wasn't wasted - the composite features (haemorrhage proxy, exudate proxy, texture score) can be used as auxiliary inputs alongside CNN embeddings in a hybrid model if the pure CNN approach plateaus.

## camera feed improvements

The current webcam capture is demonstration only - there's no way around the fact that a laptop webcam isn't a fundus camera. Two realistic upgrades:

First, add a USB slit-lamp adapter integration. These exist as consumer devices (Heine, Welch Allyn) and produce actual fundus-quality images. The camera feed code would need to detect the device and adjust capture parameters.

Second, even without fundus hardware, the eye tracking can be made more useful. Right now it captures the full frame. A better version would use the iris landmarks to estimate the centre of the eye, apply a circular crop, enhance contrast with CLAHE, and send that to the model. Still not a fundus image but noticeably better input quality.

## AR/VR integration

The most interesting direction. The idea is to run the screening inside a VR headset - the user looks into the lens, the headset camera captures a close-up of the eye at a controlled distance and lighting and the model runs inference on that image.

Meta Quest 3 has passthrough cameras at 18ppd resolution - not fundus quality but significantly better than a laptop webcam at a controlled 5cm distance. Building a Quest 3 app in Unity that captures the eye region, sends it to the Flask backend over local WiFi, and displays the result in the headset is achievable.

Longer term - a custom lens attachment for the Quest that approximates fundus optics. This has been done in research settings. Combined with a CNN model, this could be a genuinely useful field screening tool for rural clinics.

## grading explanation

Right now the model returns a grade and a confidence score. What it doesn't do is explain why. Adding a SHAP-based explanation layer would let the output say something like "Severe DR - driven primarily by high haemorrhage proxy score and elevated red channel dominance in the central region." That makes the tool significantly more useful for training junior clinicians and for building trust with referring doctors.

## multi-disease expansion

The current model only grades DR. The APTOS dataset doesn't include glaucoma or AMD labels, but other public datasets do - ORIGA for glaucoma, iChallenge-AMD for macular degeneration. A multi-task model that screens for all three from a single fundus image is the right architecture - shared CNN backbone with three separate classification heads.

## deployment

The current setup requires running a local Flask server. For clinic use that's fine - no internet dependency, patient data stays local. But a Progressive Web App (PWA) version that bundles everything including the model (ONNX export for browser inference) would remove the setup requirement entirely. A nurse could run it from an iPad without installing anything.

## what I'm not going to do

Build a cloud API version. Patient retinal images are sensitive health data. Sending them to a third-party server introduces compliance issues (HIPAA, DPDP Act in India) that aren't worth the convenience gain when local inference works fine.
