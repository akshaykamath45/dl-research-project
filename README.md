# 🕊️ Bird Species Classification Project

# Project Overview
This project aims to classify bird species using deep learning techniques. Our dataset includes 400 bird species with separate training, validation, and test sets. We are implementing and comparing various models to achieve the best classification performance.

## 📝 Current Status
### Completed Models
1. **VGG16 (With Augmentation) - by Karitkey**
2. **Inception V3 - by Kartikey**
3. **Xception - by Kartikey**
4. **EfficientNet B0 - By Akshay**
5. **VGG19 - By Sridhar Sir**

### Models Currently in Progress
6. **Custom CNN - Akshay**
7. **ResNet 50 - Sridhar Sir**

### Remaining Models to Implement
8. Transformer Model (Swine)

If any issues, we can try extra:

10. CNN without Augmentation  
11. VGG16 without Augmentation

## ✅ Important Checklist
- [ ] Upload the trained model files to GitHub.
- [ ] Create a `predict` function that allows users to select a model from available options.
- [ ] Ensure that corresponding predictions, statistics, and charts are displayed as output.
- [ ] Create frontend on streamlit
- [ ] Start with research paper documentation.

## Results Update  
After training the model, update the results in [this Google Doc](https://docs.google.com/document/d/1IUdKqsk9g5wEijbWiCzT49-3nQKj0M20Q0oy7t6vYPU/edit?tab=t.0).



---

## 🛠️ How to Train a New Model
### Step 1: Use Existing Code as Reference
- Refer to the implementation of **VGG16** or **Inception V3** in the repository.
- Adjust the model architecture and hyperparameters based on the model to be implemented.

### Step 2: Get Assistance
- Use prompts in tools like Claude or GPT with the existing code snippets to guide you.
- Ask for guidance on architecture or augmentation tweaks.

### Step 3: Modify Augmentations/Parameters
- Experiment with augmentation techniques, learning rates, and optimizers as necessary.
- Feel free to fine-tune the parameters for optimal performance.

### Step 4: Train the Model
- Use **Kaggle Notebooks** for training:
  1. Visit [this private dataset link](https://www.kaggle.com/datasets/5bc6d82a2bd2ac97d7362a0f2e8b3a19e4ff882b6c112fef14f2de1d82b5c1fe).
  2. Click **Create Notebook**.
  3. Add the required dependencies and start training.

---

## ⬛ Comparison Table

| Sr. No | Model Name          | Number of Parameters | Input Size  | Training Time              | Data Augmentation Technique Used                                                                 |
|--------|---------------------|-----------------------|-------------|----------------------------|-------------------------------------------------------------------------------------------------|
| 1      | VGG16              | ~15.9M               | 224 x 224   | ~9 hours (10 epochs, 720s/epoch) | Rotation, width/height shift, horizontal flip, zoom, shear, brightness adjustment                |
| 2      | Inception V3       | ~23.9M               | 299 x 299   | ~4 hours (25 epochs)       | Rotation, zoom, brightness, shear, flips                                                       |
| 3      | Xception           | ~22.9M               | 224 x 224   | ~4 hours (50 epochs)       | Rotation (20°), width & height shift (0.2), horizontal flip                                   |
| 4      | EfficientNet B0    | ~5.3M                | 299 x 299   | ~3.6 hours (10 epochs)     | Rescale, rotation range (20°), width shift (0.3), height shift (0.3), horizontal flip, vertical flip, zoom range (0.3), shear range (0.3), brightness range ([0.8, 1.2]), fill mode ('nearest') |
| 5      | Custom CNN         | ~14.5M               | 256 x 256   |                            | Rotation, width and height shifts, shear, zoom, horizontal flip, brightness adjustment, and nearest neighbor fill mode |
| 6      | ResNet 50          |                      |             |                            |                                                                                                 |
| 7      | Transformer Model (Swine) |               |             |                            |                                                                                                 |
| 8      | VGG19              | ~20.9M               | 224 x 224   | ~6 hours (30 epochs, ~720s/epoch)| Rotation (30°), width/height shift (0.2), horizontal flip, vertical flip, zoom range (0.2), shear range (0.2), brightness range (0.8-1.2), nearest neighbor fill mode |

---

## 📊 Comparison Chart

| Sr. No | Model Name          | Accuracy | Precision | Recall    | F1 Score  |
|--------|---------------------|----------|-----------|-----------|-----------|
| 1      | VGG16              | 0.9440   | 0.9517    | 0.9440    | 0.9421    |
| 2      | Inception V3       | 0.9695   | 0.9882    | 0.9855    | 0.9851    |
| 3      | Xception           | 0.9900   | 0.9900    | 0.9900    | 0.9900    |
| 4      | EfficientNet       | 0.9710   | 0.9772    | 0.9635    | 0.9703    |
| 5      | Custom CNN         |          |           |           |           |
| 6      | ResNet             |          |           |           |           |
| 7      | Transformer Model (Swine) |   |           |           |           |
| 8      | VGG19              | 0.9905   | 0.9673    | 0.825     | 0.8935    |


---

## 🛡️ Error Analysis
To be updated once all models are implemented and trained.





