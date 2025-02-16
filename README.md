# 🕊️ Bird Species Classification Project

# Project Overview
This project aims to classify bird species using deep learning techniques. Our dataset includes 400 bird species with separate training, validation, and test sets. We are implementing and comparing various models to achieve the best classification performance.

## 📝 Current Status

## Completed Models with Standardised Epochs and Augmentation

### Completed Models
1. **Swin Transformer**
2. **Inception V3**
3. **Xception**
4. **EfficientNet B0**
5. **VGG19**
6. **Custom CNN**
7. **GoogLeNet**




## ✅ Important Checklist
- [x] Upload the trained model files to GitHub.
- [x] Create a `predict` function that allows users to select a model from available options.
- [x] Ensure that corresponding predictions, statistics, and charts are displayed as output.
- [x] Create frontend on streamlit
- [ ] Start with research paper documentation.


## Results Update  
- After training the model, update the results in [this Google Doc](https://docs.google.com/document/d/1IUdKqsk9g5wEijbWiCzT49-3nQKj0M20Q0oy7t6vYPU/edit?tab=t.0).
- Upload the trained model files (.h5/.pth) over [here](https://drive.google.com/drive/folders/1jjB9ASm4uOFqP6APKGtxZtFJ380_eRde )


## 🛠️ Local Setup Instructions

### Method 1: Using Virtual Environment (Recommended)

1. **Clone the repository**
   ```bash
   git clone https://github.com/akshaykamath45/dl-research-project.git
   cd dl-research-project
   ```

2. **Create and activate virtual environment**
   ```bash
   # Windows
   python -m venv myenv
   myenv\Scripts\activate

   # Linux/macOS
   python3 -m venv myenv
   source myenv/bin/activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download model files**
   - Download the following model files from [Google Drive](https://drive.google.com/drive/folders/1jjB9ASm4uOFqP6APKGtxZtFJ380_eRde):
     - `customcnn.h5`
     - `inceptionv3.pth`
     - `swin.pth`
   - Place these files in the `trained models` folder of the project

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

### Method 2: Direct Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/akshaykamath45/dl-research-project.git
   cd dl-research-project
   ```

2. **Install required packages globally**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download model files**
   - Download the following model files from [Google Drive](https://drive.google.com/drive/folders/1jjB9ASm4uOFqP6APKGtxZtFJ380_eRde):
     - `customcnn.h5`
     - `inceptionv3.pth`
     - `swin.pth`
   - Place these files in the `trained models` folder of the project

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

### Project Structure
```
dl-research-project/
├── trained models/          # Place downloaded model files here
│   ├── customcnn.h5
│   ├── inceptionv3.pth
│   └── swin.pth
├── app.py                   # Main application file
├── requirements.txt         # Package dependencies
└── ...
```

### Note
- Make sure you have Python 3.7+ installed on your system
- The virtual environment method (Method 1) is recommended to avoid package conflicts
- Some models are too large for GitHub, hence they need to be downloaded separately
- Ensure all model files are placed in the correct directory before running the application


---

## ⚒️ How to Train a New Model

This guide provides a standardized approach for training models in the repository, using Inception V3 in PyTorch as a reference. Follow these steps:

### Step 1: Code Structure Overview
1. Ensure your code structure mirrors the `final_code/inceptionv3` folder.
2. Incorporate essential components:
   - **Model Definition**: Adapt the classifier layer for your dataset.
   - **Data Augmentation and Loading**: Leverage standardized preprocessing and augmentations.
   - **Metrics and Logging**: Track key metrics like accuracy, loss, precision, and recall.

### Step 2: Implement Your Model
1. Customize the model architecture as required.
2. Enable training for specific layers by setting `requires_grad=True`.

### Step 3: Prepare Data
1. Organize datasets into `train`, `valid`, and `test` directories.
2. Load data using appropriate methods for image datasets.
3. Apply a consistent augmentation pipeline during preprocessing.

### Step 4: Train the Model
1. Load any pre-existing training states.
2. Configure hyperparameters, optimizer, and learning rate scheduler.
3. Train the model and save periodic checkpoints.

### Step 5: Evaluate the Model
1. Use the best checkpoint to evaluate model performance.
2. Generate detailed evaluation metrics and visualizations.

### Step 6: Save and Log Results
1. Save key outputs such as metrics, logs, and plots.
2. Document hyperparameters to ensure reproducibility.

### Notes for New Models
- Maintain the same structure across projects for consistency.
- Use a validation set to monitor training progress.
- Update class mappings if the dataset contains new classes.

---



### Code References

#### Training Epochs
- Default training duration is 100 epochs using a standardized `train_model` function.

#### Data Augmentation Pipeline
```python
train_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

---

## ⬛ Comparison Table

| Sr. No | Model Name          | Number of Parameters | Input Size  | Training Time              | Data Augmentation Technique Used                                                                 |
|--------|---------------------|-----------------------|-------------|----------------------------|-------------------------------------------------------------------------------------------------|
| 1      | VGG16              | ~15.9M               | 224 x 224   | ~9 hours (10 epochs, 720s/epoch) | Rotation, width/height shift, horizontal flip, zoom, shear, brightness adjustment                |
| 2      | Inception V3       | ~23.9M               | 299 x 299   | ~4 hours (25 epochs)       | Rotation, zoom, brightness, shear, flips                                                       |
| 3      | Xception           | ~22.9M               | 224 x 224   | ~4 hours (50 epochs)       | Rotation (20°), width & height shift (0.2), horizontal flip                                   |
| 4      | EfficientNet B0    | ~5.3M                | 299 x 299   | ~3.6 hours (10 epochs)     | Rescale, rotation range (20°), width shift (0.3), height shift (0.3), horizontal flip, vertical flip, zoom range (0.3), shear range (0.3), brightness range ([0.8, 1.2]), fill mode ('nearest') |
| 5      | Custom CNN         | ~14.5M               | 256 x 256   |                            | Rotation, width and height shifts, shear, zoom, horizontal flip, brightness adjustment, and nearest neighbor fill mode |
| 6      | Transformer Model (Swine) |               |             |                            |                                                                                                 |
| 7     | VGG19              | ~20.9M               | 224 x 224   | ~6 hours (30 epochs, ~720s/epoch)| Rotation (30°), width/height shift (0.2), horizontal flip, vertical flip, zoom range (0.2), shear range (0.2), brightness range (0.8-1.2), nearest neighbor fill mode |

---

## 📊 Comparison Chart

| Sr. No | Model Name          | Accuracy | Precision | Recall    | F1 Score  |
|--------|---------------------|----------|-----------|-----------|-----------|
| 1      | VGG16              | 0.9440   | 0.9517    | 0.9440    | 0.9421    |
| 2      | Inception V3       | 0.9695   | 0.9882    | 0.9855    | 0.9851    |
| 3      | Xception           | 0.9900   | 0.9900    | 0.9900    | 0.9900    |
| 4      | EfficientNet       | 0.9710   | 0.9772    | 0.9635    | 0.9703    |
| 5      | Custom CNN         |          |           |           |           |
| 6      | Transformer Model (Swine) |   |           |           |           |
| 7      | VGG19              | 0.9005   | 0.9673    | 0.825     | 0.8923    |


---

## 🛡️ Error Analysis
To be updated once all models are implemented and trained.





