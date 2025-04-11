# ✍️ Handwriting Style Generator

This is a simple demo app that converts typed characters (A-Z) into a handwritten-style output using an **autoencoder** trained on grayscale images of handwritten characters.

Built with:
- 🧠 PyTorch (Autoencoder model)
- 🎨 PIL for image rendering
- ⚙️ Gradio for an interactive UI

## 🚀 How It Works

1. You type a character (like "A" or "b").
2. It's rendered as a grayscale image.
3. The image is passed through a trained autoencoder:
   - **Encoder** compresses the image into a latent representation.
   - **Decoder** reconstructs it in the original handwriting style.
4. You get a new handwritten-looking version of the character!

Currently trained on 28×28 grayscale images (MNIST-style).  
🔄 Planning to retrain on higher-resolution custom data (128×128) for better quality soon.

## 🖥️ Run Locally

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/handwriting-style-generator.git
   cd handwriting-style-generator
   ```

2. Create a virtual environment & install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

3. Make sure your `autoencoder_model.pt` is in the project folder.

4. Launch the app:
   ```bash
   python app.py
   ```

## 🛠️ To Do
- Retrain the model on 128×128 handwriting images.
- Add support for full word-to-handwriting generation.
- Improve UI with style picker for different writers.

---

Made with PyTorch.