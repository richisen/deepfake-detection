import os
from youtube import video_pred
import streamlit as st
import traceback
import sys
from architectures.weights import get_weight_path

ALLOWED_VIDEO_EXTENSIONS = {'mp4'}

def allowed_file(filename, accepted_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in accepted_extensions

@st.cache_resource
def load_model_weights(models, dataset):
    try:
        if isinstance(models, list):
            return [get_weight_path(model, dataset) for model in models]
        return get_weight_path(models, dataset)
    except Exception as e:
        st.error(f"Error downloading weights: {str(e)}")
        return None

def process_video(video_path, model, dataset, threshold, frames, weights=None):
    try:
        # Handle both single model and ensemble cases
        weight_paths = load_model_weights(model, dataset)
        if weight_paths is None:
            return "Error loading model weights", -1
        
        # Create model configuration
        if isinstance(model, list) and len(model) > 1:
            model_config = {
                'type': 'ensemble',
                'models': model,
                'weights': weights,
                'weight_paths': weight_paths
            }
        else:
            model_config = {
                'type': 'single',
                'model': model[0] if isinstance(model, list) else model,
                'weight_path': weight_paths[0] if isinstance(weight_paths, list) else weight_paths
            }
            
        output_string, pred = video_pred(
            video_path=video_path,
            model=model_config,
            dataset=dataset,
            threshold=threshold,
            frames=frames
        )
        return output_string, pred

    except Exception as e:
        return traceback.print_exception(*sys.exc_info()), -1

    finally:
        if video_path and os.path.exists(video_path):
            os.remove(video_path)