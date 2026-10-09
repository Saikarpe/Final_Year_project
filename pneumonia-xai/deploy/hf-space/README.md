---
title: PneumoScan AI
emoji: 🫁
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 5000
pinned: false
short_description: Explainable pneumonia detection on chest X-rays (research)
---

# PneumoScan AI — explainable pneumonia detection

A chest X-ray pneumonia classifier (DenseNet121) used as a testbed for
explainability: seven explanation methods, split conformal prediction with
an abstention policy, and a dashboard of explanation-quality metrics.

**Research and educational use only. Not a medical device and not a
substitute for diagnosis.**

- Trained only on paediatric radiographs (ages 1–5, Kermany et al.,
  CC BY 4.0). It has never seen an adult chest X-ray and has no external
  validation; on adult or out-of-scope images its output can be confidently
  wrong.
- Uploaded images are stored on this Space's temporary disk for the session
  and recorded (by hash, never by image) in a public audit log. Do not upload
  identifiable patient data.
- Runs on a free CPU: Grad-CAM takes about a second; Occlusion and SHAP take
  minutes.

Source: <https://github.com/Saikarpe/Final_Year_project>
