import streamlit as st
from api import process_video

# Set the title of your Streamlit app
st.title("Deepfake Detective")

# Upload file through Streamlit
uploaded_file = st.file_uploader("Choose a video...", type=["mp4"])

# Modified model selection to use multiselect
models_selected = st.multiselect(
    "Select Models (up to 3)",
    [
        "EfficientNetB4",
        "EfficientNetB4ST", 
        "EfficientNetAutoAttB4",
        "EfficientNetAutoAttB4ST",
        "Xception",
        "VGG16"
    ],
    default=["EfficientNetB4"],
    max_selections=3
)

# Add weight selection if multiple models chosen
weights = None
if len(models_selected) > 1:
    st.write("Adjust weights for selected models:")
    weights = []
    for model_name in models_selected:
        weight = st.slider(f"Weight for {model_name}", 0.0, 1.0, 1.0/len(models_selected))
        weights.append(weight)
    
    # Normalize weights to sum to 1
    total = sum(weights)
    weights = [w/total for w in weights]

threshold = st.slider("Select Threshold", 0.0, 1.0, 0.5)
frames = st.slider("Select Frames", 0, 100, 50)

# Display the uploaded file
if uploaded_file is not None:
    st.video(uploaded_file)

    # Check if the user wants to perform the deepfake detection
    if st.button("Check for Deepfake"):
        with open(f"uploads/{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.read())

        video_path = f"uploads/{uploaded_file.name}"

        result, pred = process_video(
            video_path, 
            model=models_selected,
            dataset="DFDC", 
            threshold=threshold, 
            frames=frames,
            weights=weights
        )

        st.markdown(
             f'''
            <style>
                .result{{
                    color: {'#ff4b4b' if result == 'fake' else '#6eb52f'};
                }}
            </style>
            <h3>The given video is <span class="result"> {result} </span> with confidence <span class="result">{pred if result == 'fake' else (1-pred):.2f}</span></h3>''', unsafe_allow_html=True)

else:
    st.info("Please upload a video file.")

# Add additional information or description about your project
st.divider()
st.markdown(
    '''

# Project Information
''')