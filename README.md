# Fake Review Detection Model

[![HuggingFace Model](https://img.shields.io/badge/🤗%20Model-Check%20on%20Hub-blue)](https://huggingface.co/debojit01/fake-review-detector)
[![Demo](https://img.shields.io/badge/🎮%20Live-Try%20Demo-red)](https://huggingface.co/spaces/debojit01/fake-review-detector-demo)

DistilBERT model achieving **99% accuracy** in detecting computer-generated product reviews, now with user feedback integration for continuous improvement.

## ✨ New Features
- **User Feedback System**: Collects misclassified samples for model improvement
- **Hugging Face Dataset Integration**: Automatically saves feedback to [dataset repo](https://huggingface.co/datasets/debojit01/fake-review-dataset)
- **Enhanced Demo**: Interactive Gradio interface with prediction verification

## 📊 Performance
| Metric     | Real | Fake |
|------------|------|------|
| Precision  | 0.98 | 0.99 |
| Recall     | 0.99 | 0.98 |
| F1-Score   | 0.99 | 0.99 |

## 🚀 Usage

### Python Inference
```python
from transformers import pipeline

detector = pipeline(
    "text-classification",
    model="debojit01/fake-review-detector"
)

result = detector("This product changed my life!")
print(result)  # {'label': 'FAKE', 'score': 0.99}
```

### Local Demo with Feedback
```bash
python -m venv venv
pip install gradio transformers datasets huggingface-hub
python app.py
```

## 🛠 Feedback Integration
The system automatically:
- Lets users flag incorrect predictions
- Saves corrected labels to Hugging Face dataset
- Formats data for retraining:
| text                             |  label |
| :--------------------------------| :----  |
| "Great product but poor battery" | "Real" |
| "Perfect in every way!"          | "Fake" |

## 🧠 Training Details
- **Architecture**: DistilBERT-base
**Dataset**: 40k reviews (50% real, 50% fake)
**Training**: 3 epochs (final train_loss: 0.0005)
**Continuous Learning**: Feedback samples added weekly

## 📂 Repository Structure
.
├── app.py                     # Gradio demo with feedback
├── requirements.txt           # Dependencies
└── README.md                  # This file

## 🤝 Contributing
1. Report false predictions via the demo
2. Submit PRs for model improvements
3. Add to our feedback dataset