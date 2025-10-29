import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import pickle

model = tf.keras.models.load_model("butterfly_cnn_model.h5")

with open("label_map.pkl", "rb") as f:
    label_map = pickle.load(f)

# ✅ Reverse mapping: index → label
idx_to_label = {v: k for k, v in label_map.items()}


# ✅ Prediction function (only class name, no confidence)
def predict_butterfly(img):
    img = img.convert("RGB")
    img = img.resize((128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    pred_idx = np.argmax(preds[0])
    class_name = idx_to_label[pred_idx]

    # Return only class name (Gradio Label expects a dict)
    return {class_name: None}


# ✅ Build Gradio UI
with gr.Blocks(title="🦋 Butterfly Classifier") as demo:
    gr.Markdown(
        """
        <h1 style='text-align: center; color: #9C27B0;'>🦋 Butterfly Species Classifier</h1>
        <p style='text-align: center; font-size: 16px;'>
        Upload an image of a butterfly, and the AI model will predict its species.<br>
        <b>Model Input:</b> 128×128 RGB image • <b>Framework:</b> TensorFlow + CNN
        </p>
        """
    )

    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            img_input = gr.Image(
                type="pil",
                label="📸 Upload Butterfly Image",
                height=300,
            )
            predict_btn = gr.Button("🔍 Classify Butterfly", variant="primary")

        with gr.Column(scale=1, min_width=300):
            output_label = gr.Label(
                num_top_classes=1,
                label="🔎 Predicted Species",
            )

    with gr.Accordion("ℹ️ Instructions", open=False):
        gr.Markdown(
            """
            1. Upload a clear butterfly image.  
            2. Wait for the model to process.  
            3. Only the top predicted species name will be shown (no confidence value).  
            4. Works best with close-up, clear photos.
            """
        )

    predict_btn.click(predict_butterfly, inputs=img_input, outputs=output_label)


# ✅ Launch App
if __name__ == "__main__":
    demo.launch(share=True)
