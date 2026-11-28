# EventLens ML Integration Package

This integration package is generated explicitly for deploying the trained **Multi-Layer Perceptron (MLP)** video event classification model into your backend services in **Windows VS Code**.

## 📂 Staged Directory Structure
```text
EventLens_ML_Integration_Package/
├── model_assets/
│   ├── eventlens_mlp.keras           # Trained Multi-Layer Perceptron model (TF/Keras)
│   ├── class_mapping.json            # Categorization string maps
│   ├── model_config.json             # Hyperparameters & layout configs
│   └── feature_extraction_config.json # Sampling and preprocessing parameters
├── eventlens_ml_engine.py            # EventLensMLEngine implementation
├── test_ml_pipeline.py               # Standalone test verification script
└── requirements_ml.txt               # Required Python dependencies
```

---

## 🛠️ Installation & Virtual Environment Setup

To prevent dependency conflicts within your local Windows workspace, set up a dedicated virtual environment inside your backend workspace:

```bash
# 1. Open your project backend terminal in VS Code
cd backend

# 2. Create a clean virtual environment
python -m venv venv_ml

# 3. Activate the environment (Windows PowerShell)
.\venv_ml\Scripts\Activate.ps1

# 4. Upgrade pip and install package requirements
pip install --upgrade pip
pip install -r requirements_ml.txt
```

---

## 🧪 Running Standalone Verification

You can verify the feature extraction pipeline (including uniform sampling, MobileNetV2 preprocessing, and statistical temporal aggregation) and MLP inference using the standalone verification script:

```bash
python test_ml_pipeline.py --video "path/to/your/verification_sample.avi" --threshold 0.75
```

---

## 🧠 Classification Design & Boundary Rules
- **Pre-trained Extraction**: Generates a unified vector representation containing exactly **5120 dimensions** (using 8 uniformly sampled frames at 160x160 pixels processed via MobileNetV2 backbone, calculating spatial Mean/Std and temporal Mean/Std).
- **Active Class Support**: Maps predictions to active categories including `Cultural`, `Dance`, and `Presentation`.
- **Fallback Safeguard**: If prediction probability falls below the confidence threshold (`0.75`), the class status is designated as `Unknown` due to `uncertain / low confidence`. Unsupported classes are handled systematically without crash failures.
