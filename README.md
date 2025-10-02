# Opt. BRAVO-T1

A machine learning project that classifies images of sports celebrities using facial recognition, Haar Cascade detection, and wavelet transform features.

## Quick Start

### 1. Clone the Repository
```bash
git clone <repository-url>
cd ml-uni
```

### 2. Set Up the Model Environment
```bash
cd model
python -m venv venv
```

**Activate the virtual environment:**
- Windows: `venv\Scripts\activate`
- Linux/Mac: `source venv/bin/activate`

**Install dependencies:**
```bash
pip install -r requirements.txt
```

### 3. Train the Model (Optional)
Open and run the Jupyter notebook:
```bash
jupyter notebook classifier.ipynb
```

The notebook will:
- Process images from the `dataset` folder
- Apply face and eye detection
- Extract wavelet transform features
- Train the classification model
- Save the model to `saved_model.pkl`

### 4. Set Up the Server
```bash
cd ../server
```

**Install Flask:**
```bash
pip install flask
```

**Start the Flask server:**
```bash
python server.py
```

The server will start on `http://localhost:5000`

### 5. Run the Frontend
Open `UI/app.html` in your web browser, or serve it using a local server:

```bash
cd ../UI
python -m http.server 8000
```

Then navigate to `http://localhost:8000/app.html`

### 6. Test the Application
1. Upload an image of a sports celebrity
2. Click "Analyze Image"
3. View the classification results with confidence scores

## Supported Athletes
- Cristiano Ronaldo (Football)
- Gukesh Dommaraju (Chess)
- Neeraj Chopra (Javelin)
- PV Sindhu (Badminton)
- Shubman Gill (Cricket)

## Project Structure
```
ml-uni/
├── model/              # Training data and model files
├── server/             # Flask backend
├── UI/                 # Web frontend
└── SUMMARY.md          # Technical documentation
```

## Technologies Used
- Python (Flask, OpenCV, PyWavelets, scikit-learn)
- JavaScript (Dropzone.js)
- HTML/CSS (Bootstrap)
- Machine Learning (SVM Classifier)

For detailed technical information about the image processing pipeline and feature extraction, see [SUMMARY.md](SUMMARY.md).
