import cv2
import numpy as np
import torch
from torchvision import models
 
def detect_ai_artifacts(image):
    ##Check for unnatural edges/smoothness (common in AI images).
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)      #convert coloured image to black and white
    edges = cv2.Canny(gray, 100, 200)                   #detect sharp intensity changes
    edge_score = np.mean(edges)                          # find average edge strength
    return edge_score < 30                               # a Lower edge_score = smoother (likely AI)

class ImageProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = self._load_image()
    
    def _load_image(self):
        """Load image using OpenCV with error handling."""
        try:
            image = cv2.imread(self.image_path)
            if image is None:
                raise FileNotFoundError
            return image
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def preprocess(self, target_size=(224, 224)):
        """Resize and normalize image for ML model."""
        resized = cv2.resize(self.image, target_size)
        normalized = resized / 255.0  # Scale pixel values
        return np.expand_dims(normalized, axis=0)  # Add batch dimension
    

    def is_ai(self, detector):
        """Combine ML and manual checks."""
        ml_prediction = detector.predict(self.preprocess())
        artifact_check = detect_ai_artifacts(self.image)
    
        # Return True if either method suggests AI
        return ml_prediction == "AI" or artifact_check


processor = ImageProcessor("C:/Users/tette/OneDrive/Documents/AI_Image_Detection/image2.jpg")
processed_img = processor.preprocess()
print(processed_img.shape)  # Should print (1, 224, 224, 3)

class AIDetector:
    def __init__(self):
        self.model = models.resnet18(pretrained=True)
        self.model.eval()
    
    def predict(self, input_tensor):
        #Use feature-based approach instead of classification.
        with torch.no_grad():
            tensor = torch.from_numpy(input_tensor).float()
            tensor = tensor.permute(0, 3, 1, 2)
            
            # Get features from the model (before final classification)
            features = self.model.avgpool(self.model.layer4(
                self.model.layer3(self.model.layer2(self.model.layer1(
                    self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(tensor))))
                )))
            )).flatten()
            
            # Use feature variance as AI indicator
            # AI images often have more uniform features
            feature_variance = torch.var(features).item()
            
            return "AI" if feature_variance < 0.5 else "Real"


