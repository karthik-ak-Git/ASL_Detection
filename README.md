# ASL Detection - American Sign Language Recognition System

A comprehensive deep learning solution for real-time American Sign Language (ASL) recognition using computer vision and PyTorch. This system can detect and classify ASL gestures from images, live camera feeds, and batch processing.

## 🌟 Features

- **Real-time ASL Recognition**: Live camera detection using OpenCV
- **Multiple Input Methods**: Support for single images, folder batch processing, and URL-based inference
- **Web Interface**: User-friendly frontend for easy interaction
- **REST API**: FastAPI-based API for integration with other applications
- **Pre-trained ResNet18**: Transfer learning approach for efficient training
- **Comprehensive Evaluation**: Built-in evaluation scripts with detailed metrics
- **Logging System**: Complete logging for training and inference activities
- **Modular Architecture**: Clean, organized codebase with utility functions

## 🏗️ Project Structure

```
ASL_Detection/
├── data/                    # Data processing utilities
├── dataset/                 # Training and validation datasets
├── frontend/                # Web interface (HTML, CSS, JS)
├── logs/                    # Training and application logs
├── outputs/                 # Model checkpoints and class mappings
├── utils/                   # Utility functions and helpers
├── config.py               # Configuration settings
├── main.py                 # Main training script
├── main_api.py             # FastAPI application
├── main_api_new.py         # Enhanced API version
├── live_camera.py          # Real-time camera detection
├── predict_image.py        # Single image prediction
├── predict_folder.py       # Batch folder processing
├── predict_urls.py         # URL-based predictions
├── evaluate.py             # Model evaluation
├── evaluate_new.py         # Enhanced evaluation metrics
├── feedback_trainer.py     # Interactive training with feedback
├── test_basic.py           # Basic functionality tests
├── requirements.txt        # Python dependencies
└── American Sign Language Detection.pdf  # Project documentation
```

## 🛠️ Tech Stack

- **Deep Learning**: PyTorch, TorchVision
- **Computer Vision**: OpenCV, PIL
- **Web Framework**: FastAPI, Uvicorn
- **Frontend**: HTML, CSS, JavaScript
- **Data Processing**: NumPy, Scikit-learn
- **Visualization**: Matplotlib
- **Utilities**: tqdm, python-multipart, pydantic

## 📋 Requirements

### System Requirements
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Webcam (for live detection)

### Python Dependencies
```
torch>=2.3
torchvision>=0.18
fastapi>=0.111
uvicorn[standard]>=0.30
python-multipart>=0.0.9
pydantic>=2.6
opencv-python>=4.9
numpy>=1.26
pillow>=10.2
scikit-learn>=1.5
matplotlib>=3.8
tqdm
```

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/karthik-ak-Git/ASL_Detection.git
cd ASL_Detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Your Dataset
- Place your ASL dataset in the `dataset/` directory
- Organize images in subdirectories by class (e.g., `dataset/A/`, `dataset/B/`, etc.)

### 4. Train the Model
```bash
python main.py
```

### 5. Run Real-time Detection
```bash
python live_camera.py
```

## 📚 Usage Guide

### Training the Model
The main training script uses ResNet18 with transfer learning:

```bash
python main.py
```

**Key Features:**
- Automatic GPU/CPU detection
- Data augmentation and normalization
- Learning rate scheduling
- Best model checkpointing
- Comprehensive logging

### Web API Server
Start the FastAPI server for web-based predictions:

```bash
# Basic API
uvicorn main_api:app --reload --port 8000

# Enhanced API
uvicorn main_api_new:app --reload --port 8000
```

Access the interactive API documentation at `http://localhost:8000/docs`

### Real-time Camera Detection
Run live ASL detection using your webcam:

```bash
python live_camera.py
```

### Single Image Prediction
Predict ASL gestures from individual images:

```bash
python predict_image.py --image path/to/your/image.jpg
```

### Batch Processing
Process multiple images in a folder:

```bash
python predict_folder.py --folder path/to/image/folder
```

### URL-based Predictions
Process images from URLs:

```bash
python predict_urls.py --urls urls.txt
```

### Model Evaluation
Evaluate model performance on test data:

```bash
# Basic evaluation
python evaluate.py

# Enhanced evaluation with detailed metrics
python evaluate_new.py
```

## 📁 Detailed Component Description

### Core Scripts

- **`config.py`**: Central configuration management
  - Directory paths (data, outputs, logs, frontend)
  - Model paths and class mappings
  - API settings and device configuration

- **`main.py`**: Complete training pipeline
  - ResNet18 model setup with transfer learning
  - Data loading and preprocessing
  - Training loop with validation
  - Model checkpointing and logging

- **`live_camera.py`**: Real-time detection system
  - OpenCV camera integration
  - Live inference with visual feedback
  - Real-time gesture recognition

### API Components

- **`main_api.py`**: FastAPI web server
  - RESTful endpoints for predictions
  - File upload support
  - JSON response formatting

- **`main_api_new.py`**: Enhanced API version
  - Additional features and improvements
  - Better error handling
  - Extended functionality

### Prediction Tools

- **`predict_image.py`**: Single image inference
- **`predict_folder.py`**: Batch processing utility
- **`predict_urls.py`**: URL-based image processing

### Evaluation & Testing

- **`evaluate.py`**: Model performance assessment
- **`evaluate_new.py`**: Advanced evaluation metrics
- **`test_basic.py`**: Basic functionality tests
- **`feedback_trainer.py`**: Interactive training system

### Directory Structure

- **`data/`**: Data loading and preprocessing utilities
- **`dataset/`**: Training and validation datasets
- **`frontend/`**: Web interface components
- **`logs/`**: Application and training logs
- **`outputs/`**: Model checkpoints and class mappings
- **`utils/`**: Helper functions and utilities

## 🎯 Model Architecture

The system uses a **ResNet18** architecture with transfer learning:

- **Base Model**: Pre-trained ResNet18 from TorchVision
- **Transfer Learning**: Fine-tuned for ASL classification
- **Final Layer**: Custom fully connected layer matching the number of ASL classes
- **Optimization**: Adam optimizer with learning rate scheduling
- **Loss Function**: CrossEntropyLoss for multi-class classification

## 📊 Performance & Metrics

- **Training Monitoring**: Real-time loss and accuracy tracking
- **Validation**: Separate validation set for model evaluation
- **Best Model Selection**: Automatic saving of best-performing model
- **Comprehensive Logging**: Detailed logs for analysis and debugging

## 🌐 Web Interface

The project includes a complete web frontend:
- **HTML/CSS/JavaScript**: User-friendly interface
- **Real-time Predictions**: Upload images and get instant results
- **API Integration**: Seamless connection with FastAPI backend

## 🔧 Configuration

Customize the system through `config.py`:

```python
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dataset"
OUTPUTS_DIR = ROOT / "outputs"
LOGS_DIR = ROOT / "logs"
FRONTEND_DIR = ROOT / "frontend"
MODEL_PATH = OUTPUTS_DIR / "best_model.pth"
DEFAULT_DEVICE = "cuda"  # fallback to cpu if not available
```

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size in data loaders
   - Use CPU by setting `DEVICE = "cpu"` in config

2. **Camera Access Issues**:
   - Check camera permissions
   - Ensure OpenCV is properly installed
   - Try different camera indices in `live_camera.py`

3. **Module Import Errors**:
   - Verify all dependencies are installed
   - Check Python path configuration
   - Ensure you're in the correct directory

4. **Model Loading Errors**:
   - Train the model first using `main.py`
   - Check if model files exist in `outputs/` directory
   - Verify class mapping files are generated

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the Repository**
2. **Create a Feature Branch**: `git checkout -b feature/amazing-feature`
3. **Commit Changes**: `git commit -m 'Add amazing feature'`
4. **Push to Branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Areas for Contribution
- Additional ASL gesture classes
- Model architecture improvements
- Web interface enhancements
- Performance optimizations
- Documentation improvements
- Bug fixes and testing

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**Karthik AK**
- GitHub: [@karthik-ak-Git](https://github.com/karthik-ak-Git)
- Project Link: [https://github.com/karthik-ak-Git/ASL_Detection](https://github.com/karthik-ak-Git/ASL_Detection)

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- TorchVision for pre-trained models
- OpenCV community for computer vision tools
- FastAPI developers for the modern web framework
- ASL community for inspiration and support

## 📈 Future Enhancements

- [ ] Support for additional sign languages
- [ ] Mobile app development
- [ ] Real-time sentence formation
- [ ] Voice synthesis integration
- [ ] Cloud deployment options
- [ ] Model quantization for edge devices
- [ ] Multi-hand gesture recognition
- [ ] Temporal sequence modeling for dynamic gestures

---

⭐ **Star this repository if you find it helpful!**

For questions, issues, or suggestions, please open an issue on GitHub.
