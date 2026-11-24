
EVENTLENS ACTION RECOGNITION MODEL

Classifier: StandardScaler + LinearSVC
Feature extractor: R3D-18 KINETICS400_V1
Feature dimensions: 512
Frames per clip: 16
Resolution: 112x112

Classes: 12 UCF101 action categories.

The classifier predicts UCF101 action categories.
It is not an independently trained event-level classifier.

Required Python packages:
torch, torchvision, scikit-learn, joblib, opencv-python
