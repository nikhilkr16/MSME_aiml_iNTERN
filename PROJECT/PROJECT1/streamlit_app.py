import streamlit as st
import os
from utils import PlantDiseasePredictor

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
            
            # Display top predictions
            st.subheader("Top Predictions:")
            for disease, conf in result['top_3_predictions']:
                st.write(f"- {disease.replace('_', ' ')}: {conf:.2%}")
            
            # Disease info
            disease_info = predictor.get_disease_info(predicted_class)
            st.subheader("🔬 Disease Information")
            st.write(f"**Description:** {disease_info['description']}")
            st.write(f"**Treatment:** {disease_info['treatment']}")

if __name__ == "__main__":
    main() 
