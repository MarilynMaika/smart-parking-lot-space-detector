import gradio as gr
import pickle
import numpy as np
from skimage.transform import resize

# Load your trained GridSearchCV SVC model
model_path = os.path.join("smart_parking_lot_space_detector", "model.p")
with open(model_path, "rb") as file:
    model = pickle.load(file)

def predict_parking_space(img):
    # Convert PIL image to numpy array
    img = np.array(img)
    
    # Preprocess exactly as during training
    img = resize(img, (15, 15))
    img = img[:, :, :3]  # ensure 3 channels (RGB)
    
    # Flatten to (1, 675)
    features = img.flatten().reshape(1, -1)
    
    # Predict
    pred = model.predict(features)[0]
    
    # Generate label + color + emoji
    if pred == 0:
        label = "✅ Empty"
        color = "green"
    else:
        label = "🚗 Not Empty"
        color = "red"
    
    # Return as HTML for colored output
    return f"<span style='color:{color}; font-size: 1.5em;'>{label}</span>"

# Create Gradio interface
app = gr.Interface(
    fn=predict_parking_space,
    inputs=gr.Image(type="pil", label="Upload Parking Space Image"),
    outputs=gr.HTML(label="Prediction"),
    title="🚘 Parking Lot Space Classifier",
    description="A smart parking space detector that helps drivers and parking managers quickly identify available spots in real time..",

)

app.launch(share=True)
