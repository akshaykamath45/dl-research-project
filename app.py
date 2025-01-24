import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import json
import numpy as np
from torchvision import models
import timm
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch.nn as nn
import tensorflow as tf

# Set page config
st.set_page_config(page_title="Bird Species Classification", layout="wide")

# Custom CSS to improve the look
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        background-color: #ffffff;
        border: 2px solid #4CAF50;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-box h3 {
        color: #2c3e50;
        margin-bottom: 10px;
        font-size: 1.2em;
    }
    .prediction-box h2 {
        color: #4CAF50;
        margin: 10px 0;
        font-size: 1.8em;
        font-weight: bold;
    }
    .prediction-box p {
        color: #34495e;
        font-size: 1.1em;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Load class names based on model
@st.cache_resource
def load_class_mapping(model_name):
    if model_name in ['vgg19', 'customcnn']:  # Add customcnn to use the same mapping as VGG19
        # For VGG19 and CustomCNN, create a mapping of 400 classes (0-399)
        return {str(i): f"Bird Species {i+1}" for i in range(400)}
    else:
        # For PyTorch models, load from their respective results folders
        mapping_path = os.path.join('final code', model_name, 'results', 'class_mapping.json')
        try:
            with open(mapping_path, 'r') as f:
                mapping = json.load(f)
                # Convert keys to strings to ensure consistent key types
                return {str(k): v for k, v in mapping.items()}
        except FileNotFoundError:
            st.error(f"Class mapping file not found: {mapping_path}")
            return None
        except Exception as e:
            st.error(f"Error loading class mapping: {str(e)}")
            return None

# Model name mapping
MODEL_MAPPING = {
    'inceptionv3': 'inceptionv3',
    'swin_b': 'swin_b',
    'xception': 'xception',
    'efficientnet_b0': 'efficientnetb0',
    'vgg19': 'vgg19',
    'googlenet': 'googlenet',
    'customcnn': 'customcnn'  # Add custom CNN model
}

# Model filename mapping
MODEL_FILE_MAPPING = {
    'inceptionv3': 'inceptionv3.pth',
    'swin_b': 'swin.pth',
    'xception': 'xception.pth',
    'efficientnet_b0': 'effecientnet.pth',
    'vgg19': 'vgg19.h5',
    'googlenet': 'googlenet.pth',
    'customcnn': 'customcnn.h5'  # Add custom CNN model file
}

class BirdClassifier(nn.Module):
    def __init__(self, num_classes):
        super(BirdClassifier, self).__init__()
        self.inception = models.inception_v3(weights='DEFAULT')
        
        # Selectively unfreeze later layers
        for param in self.inception.parameters():
            param.requires_grad = False
            
        # Unfreeze the last two Mixed layers
        for layer in [self.inception.Mixed_7a, self.inception.Mixed_7b, self.inception.Mixed_7c]:
            for param in layer.parameters():
                param.requires_grad = True
            
        # Replace the final layer with improved architecture
        num_features = self.inception.fc.in_features
        self.inception.fc = nn.Sequential(
            nn.Linear(num_features, 1536),
            nn.BatchNorm1d(1536),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1536, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 400)  # 400 classes
        )
        
    def forward(self, x):
        if self.training:
            logits, aux_logits = self.inception(x)
            return logits, aux_logits
        return self.inception(x)

class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetClassifier, self).__init__()
        # Initialize without pre-trained weights
        self.efficientnet = models.efficientnet_b0(weights=None)
        
        # Freeze early layers
        for param in self.efficientnet.parameters():
            param.requires_grad = False
            
        # Unfreeze the last few layers
        for layer in [self.efficientnet.features[-1], self.efficientnet.classifier]:
            for param in layer.parameters():
                param.requires_grad = True
            
        # Replace the final layer with improved architecture
        num_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, num_classes)
        )
        
    def forward(self, x):
        return self.efficientnet(x)

class XceptionClassifier(nn.Module):
    def __init__(self, num_classes):
        super(XceptionClassifier, self).__init__()
        self.xception = timm.create_model('xception', pretrained=True)
        
        # Freeze all layers
        for param in self.xception.parameters():
            param.requires_grad = False
            
        # Replace classifier with custom architecture
        num_features = self.xception.fc.in_features  # This is 2048 for Xception
        self.xception.fc = nn.Sequential(
            nn.Linear(num_features, 1536),
            nn.BatchNorm1d(1536),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1536, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 400)  # 400 classes
        )
        
    def forward(self, x):
        return self.xception(x)

class SwinClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SwinClassifier, self).__init__()
        self.swin = models.swin_b(weights='DEFAULT')
        
        # Freeze all parameters initially
        for param in self.swin.parameters():
            param.requires_grad = False
            
        # Unfreeze the last two stages
        for layer in [self.swin.features[-1], self.swin.features[-2]]:
            for param in layer.parameters():
                param.requires_grad = True
        
        # Get the number of features from the last layer
        num_features = self.swin.head.in_features
        
        # Replace the classification head with improved architecture
        self.swin.head = nn.Sequential(
            nn.Linear(num_features, 1536),
            nn.BatchNorm1d(1536),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1536, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )
        
    def forward(self, x):
        return self.swin(x)

class GoogLeNetBirdClassifier(nn.Module):
    def __init__(self, num_classes):
        super(GoogLeNetBirdClassifier, self).__init__()
        self.googlenet = models.googlenet(weights='DEFAULT')
        
        # Freeze all parameters initially
        for param in self.googlenet.parameters():
            param.requires_grad = False
            
        # Unfreeze the last inception blocks (a3, b3)
        layers_to_unfreeze = ['inception5a', 'inception5b']
        for name, param in self.googlenet.named_parameters():
            if any(layer in name for layer in layers_to_unfreeze):
                param.requires_grad = True
        
        # Modified classifier head
        num_features = self.googlenet.fc.in_features
        self.googlenet.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.googlenet(x)

# Define model loading functions for PyTorch models
@st.cache_resource
def load_pytorch_model(model_name):
    if model_name == 'inceptionv3':
        model = BirdClassifier(num_classes=400)
    elif model_name == 'swin_b':
        model = SwinClassifier(num_classes=400)
    elif model_name == 'xception':
        model = XceptionClassifier(num_classes=400)
    elif model_name == 'efficientnet_b0':
        model = EfficientNetClassifier(num_classes=400)
    elif model_name == 'googlenet':
        model = GoogLeNetBirdClassifier(num_classes=400)
    
    # Load the trained weights using the correct filename
    model_path = os.path.join('trained models', MODEL_FILE_MAPPING[model_name])
    try:
        # Load state dict with map_location and weights_only=True
        state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
        
        # Handle potential DataParallel wrapper
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # Load state dict
        model.load_state_dict(state_dict, strict=False)
        model.eval()  # Set to evaluation mode
        return model
    except FileNotFoundError:
        st.error(f"Model file not found: {model_path}")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Image transformation functions
def transform_image_pytorch(image, model_name):
    if model_name == 'inceptionv3':
        size = 299
    elif model_name in ['swin_b', 'efficientnet_b0', 'googlenet']:
        size = 224
    elif model_name == 'xception':
        size = 299
    else:  # default
        size = 224
        
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    return transform(image)

def transform_image_tensorflow(image, model_name='vgg19'):
    # Resize to correct input size
    image = image.resize((224, 224))
    # Convert to numpy array and preprocess
    img_array = np.array(image)
    
    if model_name == 'vgg19':
        # VGG19 specific preprocessing
        img_array = tf.keras.applications.vgg19.preprocess_input(img_array)
    else:  # Custom CNN preprocessing
        # Rescale to [0,1] as per the training code
        img_array = img_array / 255.0
    
    return np.expand_dims(img_array, axis=0)

# Prediction function
def predict_image(model, image, model_type, model_name):
    if model_type == 'tensorflow':
        transformed_image = transform_image_tensorflow(image, model_name)
        predictions = model.predict(transformed_image, verbose=0)  # Set verbose=0 to avoid printing prediction messages
        class_probs = predictions[0]
    else:
        transformed_image = transform_image_pytorch(image, model_name)
        with torch.no_grad():
            transformed_image = torch.unsqueeze(transformed_image, 0)
            outputs = model(transformed_image)
            class_probs = torch.softmax(outputs, dim=1)[0].numpy()
    
    return class_probs

# Plot confidence distribution
def plot_confidence_distribution(predictions, class_names, top_k=5):
    top_k_idx = np.argsort(predictions)[-top_k:][::-1]
    top_k_values = [float(predictions[idx]) for idx in top_k_idx]  # Convert to Python float
    top_k_classes = [class_names[str(int(idx))] for idx in top_k_idx]  # Convert indices to string for mapping
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(top_k_classes, top_k_values)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2%}',
                ha='center', va='bottom')
    
    plt.xticks(rotation=45, ha='right')
    plt.title('Top 5 Predictions Confidence Distribution')
    plt.tight_layout()
    
    return fig

# Main Streamlit app
def main():
    st.title("🦜 Bird Species Classification")
    st.write("Upload an image of a bird and select a model to classify the species")
    
    # Model selection
    model_choice = st.selectbox(
        "Select a model",
        list(MODEL_MAPPING.keys()),
        help="Choose the model you want to use for classification"
    )
    
    # Load class mapping for selected model
    class_mapping = load_class_mapping(MODEL_MAPPING[model_choice])
    if class_mapping is None:
        st.error("Failed to load class mapping. Please check if the files are in the correct location.")
        return
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload an image of a bird", 
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )
    
    if uploaded_file is not None:
        # Display the uploaded image
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
        
        # Load model and make prediction
        if st.button("Classify", type="primary"):
            with st.spinner("Processing image..."):
                try:
                    if model_choice in ['vgg19', 'customcnn']:  # Add customcnn to use tensorflow loading
                        model = load_tensorflow_model(model_choice)
                        model_type = 'tensorflow'
                    else:
                        model = load_pytorch_model(model_choice)
                        model_type = 'pytorch'
                    
                    if model is None:
                        st.error("Failed to load model. Please check if the model files are in the correct location.")
                        return
                    
                    # Get predictions
                    predictions = predict_image(model, image, model_type, model_choice)
                    
                    with col2:
                        st.subheader("Prediction Results")
                        
                        # Get top prediction
                        top_pred_idx = int(np.argmax(predictions))  # Convert to Python int
                        top_pred_class = class_mapping[str(top_pred_idx)]  # Convert to string for JSON mapping
                        top_pred_conf = float(predictions[top_pred_idx])  # Convert to Python float
                        
                        # Display top prediction with confidence
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h3>Predicted Species:</h3>
                            <h2>{top_pred_class}</h2>
                            <p>Confidence: {top_pred_conf:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display confidence distribution plot
                        st.subheader("Confidence Distribution")
                        fig = plot_confidence_distribution(predictions, class_mapping)
                        st.pyplot(fig)
                        
                        # Display model information
                        st.subheader("Model Information")
                        st.info(f"""
                        - Model: {model_choice.upper()}
                        - Type: {'TensorFlow' if model_type == 'tensorflow' else 'PyTorch'}
                        - Input Size: {224 if model_choice in ['swin_b', 'efficientnet_b0', 'vgg19', 'customcnn'] else 299} x {224 if model_choice in ['swin_b', 'efficientnet_b0', 'vgg19', 'customcnn'] else 299}
                        """)
                
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.error("Please try again with a different image or model.")

# Load TensorFlow model
@st.cache_resource
def load_tensorflow_model(model_name='vgg19'):
    model_path = os.path.join('trained models', MODEL_FILE_MAPPING[model_name])
    try:
        if model_name == 'vgg19':
            # VGG19 specific loading
            base_model = tf.keras.applications.VGG19(
                weights='imagenet',
                include_top=False,
                input_shape=(224, 224, 3)
            )
            
            # Create the model architecture
            x = base_model.output
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dense(1024, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.5)(x)
            x = tf.keras.layers.Dense(512, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.4)(x)
            predictions = tf.keras.layers.Dense(400, activation='softmax')(x)
            
            model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
            # Load weights
            model.load_weights(model_path)
        else:  # Custom CNN model
            # Create the custom CNN model architecture
            model = tf.keras.Sequential([
                # First Convolutional Block
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3), 
                                     kernel_initializer='he_normal', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.25),
                
                # Second Convolutional Block
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu', 
                                     kernel_initializer='he_normal', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu', 
                                     kernel_initializer='he_normal', padding='same'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.3),
                
                # Third Convolutional Block
                tf.keras.layers.Conv2D(128, (3, 3), activation='relu', 
                                     kernel_initializer='he_normal', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(128, (3, 3), activation='relu', 
                                     kernel_initializer='he_normal', padding='same'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.35),
                
                # Fourth Convolutional Block
                tf.keras.layers.Conv2D(256, (3, 3), activation='relu', 
                                     kernel_initializer='he_normal', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(256, (3, 3), activation='relu', 
                                     kernel_initializer='he_normal', padding='same'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.4),
                
                # Flatten and Fully Connected Layers
                tf.keras.layers.Flatten(),
                
                tf.keras.layers.Dense(512, activation='relu', 
                                    kernel_initializer='he_normal',
                                    kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.5),
                
                tf.keras.layers.Dense(256, activation='relu', 
                                    kernel_initializer='he_normal',
                                    kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.4),
                
                # Output Layer
                tf.keras.layers.Dense(400, activation='softmax')
            ])
            
            # Load weights
            model.load_weights(model_path)
            
            # Compile model
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0),
                loss='categorical_crossentropy',
                metrics=['accuracy',
                        tf.keras.metrics.Precision(),
                        tf.keras.metrics.Recall()]
            )
        return model
    except FileNotFoundError:
        st.error(f"Model file not found: {model_path}")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

if __name__ == "__main__":
    main()