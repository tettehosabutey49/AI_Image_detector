# 🔍 AI Image Detector
## 🎯 Learning Objectives  
After finishing *Think Python*, I built this project to:  
✅ **Practice class-based design** (`ImageProcessor`, `AIDetector`)  
✅ **Implement proper function decomposition** (single-responsibility functions)  
✅ **Use Pythonic patterns**:  
   - Context managers (`with torch.no_grad()`)  
   - Type-agnostic processing (NumPy/PyTorch interoperability)  
✅ **Work with computer vision libraries** (OpenCV, PIL)  
✅ **Build a deployable tool** (Streamlit UI)  

## 🚀 Features
- **Dual Detection System**:
  - Computer vision-based edge analysis
  - Deep feature extraction using ResNet18
- **Web Interface**: Easy-to-use Streamlit app
- **Batch Processing**: Handles multiple image formats
- **Threshold Customization**: Adjustable sensitivity

## 🛠️ Technical Implementation Highlights  

### 1. Class-Based Architecture  
```python
class ImageProcessor:
    def __init__(self, image_path):
        self.image_path = image_path  # Encapsulation
        self.image = self._load_image()  # Private method
    
    def _load_image(self):  # Clear separation of concerns
        """Load image with error handling."""
        try:
            image = cv2.imread(self.image_path)
            return image if image is not None else raise FileNotFoundError
        except Exception as e:
            print(f"Error: {e}")
            return None





## 📦 Installation

### Prerequisites
- Python 3.8+
- pip

### Setup
```bash
git clone https://github.com/yourusername/ai-image-detector.git
cd ai-image-detector
pip install -r requirements.txt

### To use this project, run this code in your terminal
streamlit run app.py

### Project Structure
ai-image-detector/
├── app.py                # Web interface
├── image_processor.py    # Core detection logic
├── README.md
├── requirements.txt
└── /test_images          # Sample images for testing

To contribute:

Fork the repository

Create a feature branch

Submit a pull request