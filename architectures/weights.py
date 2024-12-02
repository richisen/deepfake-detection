import os
import gdown

# Replace with your Google Drive file IDs
weight_path = {
    'EfficientNetAutoAttB4ST_DFDC': '1Eek2sorYZn7xg58_IA9NIgwm60wDHO8E',
    'EfficientNetAutoAttB4_DFDC': '1SbiTWmb2ilE9DGdQy2qyGoYq44BYkYnA',
    'EfficientNetB4ST_DFDC': '1lfwxQXkU6V-4d3EZaMAqH26XxfXw7QLH',
    'EfficientNetB4_DFDC': '1FluhVEow2_Dl2p8PP5ZvwLPteO9sUSGs',
    'Xception_DFDC': '1BX4u7rC4-ZHCv6ux7iyhD6961oZF3KyR',
    'VGG16_DFDC': '1yN9S1SxtTgwBz06um4HDgqOd3mqNKIrt'
}

def get_weight_path(model_name, dataset):
    """Download and return path to model weights"""
    file_id = weight_path[f"{model_name}_{dataset}"]
    
    # Create weights directory if it doesn't exist
    os.makedirs('weights', exist_ok=True)
    
    output_path = f"weights/{model_name}_{dataset}_bestval.pth"
    
    # Only download if file doesn't exist
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    
    return output_path