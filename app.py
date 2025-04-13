import gradio as gr
from transformers import pipeline
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi, notebook_login
import os
import pandas as pd

# Initialize detector
detector = pipeline("text-classification", model="debojit01/fake-review-detector")

# Hugging Face Dataset setup
HF_DATASET = "debojit01/fake-review-dataset"
TOKEN = os.environ.get("HF_TOKEN")  # Set this in Space secrets

def predict(text):
    result = detector(text)[0]
    if result["label"] == "LABEL_0":  # Real
        return {"Real": result["score"], "Fake": 1 - result["score"]}
    else:  # Fake (LABEL_1)
        return {"Real": 1 - result["score"], "Fake": result["score"]}

def save_feedback(text, prediction, is_correct):
    """Save feedback to HF dataset"""
    try:
        # Load existing dataset
        dataset = load_dataset(HF_DATASET)['train']
        df = dataset.to_pandas()
    except:
        df = pd.DataFrame(columns=["text", "label"])
    
    # Determine correct label
    predicted_label = "Real" if prediction["Real"] > 0.5 else "Fake"
    true_label = predicted_label if is_correct else ("Fake" if predicted_label == "Real" else "Real")
    
    # Append new data
    new_row = {"text": text, "label": true_label}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Convert back to dataset and push
    updated_dataset = Dataset.from_pandas(df)
    updated_dataset.push_to_hub(
        HF_DATASET,
        token=TOKEN,
        commit_message=f"New feedback added via app"
    )
    return "Feedback saved to dataset!"

with gr.Blocks() as app:
    gr.Markdown("## Fake Review Detector")
    
    with gr.Row():
        review_input = gr.Textbox(label="Enter Review")
        predict_btn = gr.Button("Predict")
    
    output_label = gr.Label(label="Prediction")
    
    with gr.Row(visible=False) as feedback_row:
        feedback_radio = gr.Radio(
            ["Correct", "Incorrect"],
            label="Is this prediction accurate?"
        )
        feedback_btn = gr.Button("Submit Feedback")
    
    status_text = gr.Textbox(label="Status", interactive=False)

    def show_prediction(text):
        prediction = predict(text)
        return prediction, gr.Row(visible=True), ""

    predict_btn.click(
        show_prediction,
        inputs=review_input,
        outputs=[output_label, feedback_row, status_text]
    )

    feedback_btn.click(
        save_feedback,
        inputs=[review_input, output_label, feedback_radio],
        outputs=status_text
    )

app.launch()