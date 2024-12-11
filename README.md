# 🕊️ Bird Species Classification Project

##  Project Overview
This project aims to classify bird species using deep learning techniques. Our dataset includes 400 bird species with separate training, validation, and test sets. We are implementing and comparing various models to achieve the best classification performance.

## 📝 Current Status
### Completed Models
1. **VGG16 (With Augmentation) - by Karitkey**
2. **Inception V3 - by Kartikey**

### Models Currently in Progress
3. **Custom CNN - Akshay**
4. **YOLO - Sridhar Sir**
5. **Xception - Kartikey**

### Remaining Models to Implement
6. EfficientNet B0 - Akshay
7. ResNet 50 - Sridhar Sir
8. Transformer Models (Swine) 
9. VGG19

If any issues, we can try extra

10. CNN without Augmentation
11. VGG16 without Augmentation

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

## 📊 Comparison Table
| Sr. No | Model Name    | Parameters | Input Size  | Training Time      | Data Augmentation Techniques                                                  |
|--------|---------------|------------|-------------|--------------------|-------------------------------------------------------------------------------|
| 1      | VGG16         | ~15.9M     | 224 x 224   | ~9 hours (10 epochs, 720s/epoch) | Rotation, width/height shift, horizontal flip, zoom, shear, brightness        |
| 2      | Inception V3  | ~23.9M     | 299 x 299   | ~4 hours (25 epochs) | Rotation, zoom, brightness, shear, flips                                     |
| 3      | Custom CNN    | ~14.5M     | 256 x 256   | In Progress        | Rotation, width/height shifts, shear, zoom, horizontal flip, brightness       |

---

## 📈 Comparison Chart
| Sr. No | Model Name    | Accuracy  | Precision | Recall    | F1 Score  |
|--------|---------------|-----------|-----------|-----------|-----------|
| 1      | VGG16         | 0.9440    | 0.9517    | 0.9440    | 0.9421    |
| 2      | Inception V3  | 0.9695    | 0.9882    | 0.9855    | 0.9851    |
| 3      | Custom CNN    | In Progress | In Progress | In Progress | In Progress |

---

## 🛡️ Error Analysis
To be updated once all models are implemented and trained.





