class AIDetector:
    def __init__(self):
        self.model = models.resnet18(pretrained=True)
        self.model.eval()  # Set to evaluation mode
    
    def predict(self, input_tensor):
        """Convert NumPy array to tensor and predict."""
        with torch.no_grad():
            tensor = torch.from_numpy(input_tensor).float()
            tensor = tensor.permute(0, 3, 1, 2)  # Change to (batch, channels, H, W)
            output = self.model(tensor)
        return "AI" if output.argmax().item() == 1 else "Real"

detector = AIDetector()
result = detector.predict(processed_img)
print(f"Prediction: {result}")