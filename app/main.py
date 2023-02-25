import cv2
import numpy as np
import tensorflow as tf
import uvicorn
from fastapi import FastAPI

app = FastAPI()

### Load model
model = tf.keras.models.load_model("best_model.h5")
class_map = {'Apple_Green': 0, 'Apple_Red': 1, 'Apple_Yellow': 2, 'Guava_Green': 3, 'Guava_Red': 4, 'Guava_Yellow': 5, 'Mandarine_Green': 6, 'Mandarine_Red': 7, 'Mandarine_Yellow': 8, 'Orange_Green': 9, 'Orange_Red': 10, 'Orange_Yellow': 11}
# Define a function to preprocess the image
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Define the endpoint for image classification
@app.post("/predict")
async def predict(image_path: str):
    # Read the image using OpenCV
    img = cv2.imread(rf"{image_path}")
    
    # Preprocess the image
    img = preprocess_image(img)
    
    # Make a prediction using the model
    pred = model.predict(img)
    pred_index = tf.argmax(pred, axis=1)
    label = class_map[pred_index]
    
    # Return the prediction result
    return {"prediction": label}

if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=5001, reload=True)
