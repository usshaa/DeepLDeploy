import json
import numpy as np
import onnxruntime as ort
from PIL import Image
import io

class FaceRecognitionPredictor:
    def __init__(self, 
                 model_path="model_optimized.onnx", 
                 labels_path="labels.json"):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        with open(labels_path, "r") as f:
            self.labels = json.load(f)
            
    def preprocess(self, pil_img):
        img = pil_img.convert('RGB').resize((224, 224))
        arr = np.array(img).astype(np.float32) / 255.0 #=(0-1)
        # Normalize ImageNet mean/std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        arr = np.transpose(arr, (2, 0, 1)) # HWC (224X224X3) to CHW (3X224X224)
        return np.expand_dims(arr, axis=0) # Add batch dimension (NCHW) (1,3,224,224)
# [N,C,H,W] - [1,3,224,224]
# [N,31] - [1,31]
    def predict(self, image_path):
        img = Image.open(image_path)
        tensor = self.preprocess(img)
        outputs = self.session.run(None, {self.input_name: tensor})[0]
        class_id = int(np.argmax(outputs, axis=1)[0])
        confidence = float(np.exp(outputs[0][class_id]) / np.sum(np.exp(outputs[0])))
        return {"identity": self.labels[str(class_id)], "confidence": confidence}

    def predict1(self, image_bytes):
        img = Image.open(io.BytesIO(image_bytes))
        tensor = self.preprocess(img)
        outputs = self.session.run(None, {self.input_name: tensor})[0]
        class_id = int(np.argmax(outputs, axis=1)[0])
        confidence = float(np.exp(outputs[0][class_id]) / np.sum(np.exp(outputs[0])))
        return {"identity": self.labels[str(class_id)], "confidence": confidence}

if __name__ == "__main__":
    predictor = FaceRecognitionPredictor()
    print("Inference service initialized and ready for production.")