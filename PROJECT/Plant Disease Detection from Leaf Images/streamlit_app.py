import streamlit as st
import numpy as np
from PIL import Image
import os
from utils import PlantDiseasePredictor

# Try importing optional dependencies
try:
    import plotly.express as px
    import pandas as pd
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Some visualization features are disabled. Install plotly and pandas for full functionality.")

# Page configuration
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #2e7d32;
    text-align: center;
    margin-bottom: 2rem;
}
.prediction-box {
    background-color: #f8f9fa;
    border-radius: 10px;
    padding: 1rem;
    border-left: 5px solid #4CAF50;
}
.disease-info {
    background-color: #e3f2fd;
    border-radius: 10px;
    padding: 1rem;
    border-left: 5px solid #2196F3;
}
.confidence-high {
    color: #4CAF50;
    font-weight: bold;
}
.confidence-medium {
    color: #FF9800;
    font-weight: bold;
}
.confidence-low {
    color: #F44336;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_predictor():
    """Load the plant disease predictor model"""
    return PlantDiseasePredictor()

def main():
    st.title("🌱 Plant Disease Detection System")
    
    # Load predictor
    try:
        predictor = load_predictor()
        model_loaded = predictor.model is not None
    except:
        model_loaded = False
    
    if not model_loaded:
        st.warning("⚠️ Running in DEMO mode: No real predictions will be made.")
        st.info("To use real predictions, run `python model_training.py` first.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📷 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a plant leaf image...",
            type=['jpg', 'jpeg', 'png', 'bmp']
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("🔍 Predict Disease", type="primary"):
                with st.spinner("Analyzing..."):
                    result, message = predictor.predict_disease(pil_image=image)
                    
                    if result is None:
                        st.error(f"❌ Error: {message}")
                    else:
                        st.session_state.result = result
    
    with col2:
        st.header("📊 Results")
        
        if 'result' in st.session_state:
            result = st.session_state.result
            predicted_class = result['predicted_class']
            confidence = result['confidence']
            
            st.success(f"🏆 **Predicted:** {predicted_class.replace('_', ' ')}")
            st.info(f"📊 **Confidence:** {confidence:.2%}")
            
            # Confidence chart
            if PLOTLY_AVAILABLE:
                all_preds = result['all_predictions']
                df = pd.DataFrame([
                    {'Disease': k.replace('_', ' '), 'Confidence': v * 100} 
                    for k, v in all_preds.items()
                ]).sort_values('Confidence', ascending=True)
                
                fig = px.bar(df, x='Confidence', y='Disease', orientation='h')
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Simple text-based display if plotly is not available
                st.subheader("Top Predictions:")
                for disease, conf in result['top_3_predictions']:
                    st.write(f"- {disease.replace('_', ' ')}: {conf:.2%}")
            
            # Disease info
            disease_info = predictor.get_disease_info(predicted_class)
            st.subheader("🔬 Disease Information")
            st.write(f"**Description:** {disease_info['description']}")
            st.write(f"**Treatment:** {disease_info['treatment']}")

# Additional pages
def show_model_info():
    st.header("🧠 Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Architecture")
        st.markdown("""
        - **Type**: Convolutional Neural Network (CNN)
        - **Input Size**: 224 x 224 x 3
        - **Layers**: 4 Conv2D blocks with BatchNorm
        - **Dense Layers**: 2 fully connected layers
        - **Output**: 5 disease classes
        """)
    
    with col2:
        st.subheader("Training Details")
        st.markdown("""
        - **Dataset**: PlantVillage
        - **Optimizer**: Adam
        - **Loss**: Categorical Crossentropy
        - **Metrics**: Accuracy
        - **Augmentation**: Rotation, Shift, Zoom, Flip
        """)
    
    # Model performance (if available)
    if os.path.exists('models/training_history.png'):
        st.subheader("📊 Training History")
        st.image('models/training_history.png')
    
    if os.path.exists('models/confusion_matrix.png'):
        st.subheader("🎯 Confusion Matrix")
        st.image('models/confusion_matrix.png')

def show_dataset_info():
    st.header("📊 Dataset Information")
    
    st.markdown("""
    ### PlantVillage Dataset
    
    The PlantVillage dataset is a collection of leaf images used for plant disease classification:
    
    - **Total Images**: 50,000+ images
    - **Plants**: 14 crop species
    - **Diseases**: 26 diseases + healthy plants
    - **Image Format**: RGB color images
    - **Resolution**: Various sizes (resized to 224x224 for training)
    
    ### Classes in This Model
    
    This model is trained on 5 classes:
    1. **Apple Apple Scab** - Fungal disease causing dark lesions
    2. **Apple Black Rot** - Serious fungal disease with brown/black lesions
    3. **Apple Cedar Apple Rust** - Disease causing yellow spots and orange lesions
    4. **Apple Healthy** - Healthy apple plants with no disease
    5. **Tomato Bacterial Spot** - Bacterial disease causing small dark spots
    """)

# Navigation
def show_navigation():
    pages = {
        "🏠 Home": main,
        "🧠 Model Info": show_model_info,
        "📊 Dataset Info": show_dataset_info
    }
    
    # Page selection
    if 'page' not in st.session_state:
        st.session_state.page = "🏠 Home"
    
    # Navigation in sidebar
    with st.sidebar:
        st.markdown("---")
        selected_page = st.radio("Navigation", list(pages.keys()))
        
        if selected_page != st.session_state.page:
            st.session_state.page = selected_page
            st.rerun()
    
    # Show selected page
    pages[st.session_state.page]()

if __name__ == "__main__":
    if len(st.session_state) == 0:
        show_navigation()
    else:
        show_navigation() 
