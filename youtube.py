import torch
from torch.utils.model_zoo import load_url
from scipy.special import expit
import os

import sys
sys.path.append('..')

from blazeface import FaceExtractor, BlazeFace, VideoReader
from architectures import fornet, weights
from isplutils import utils

def video_pred(threshold=0.5, model=None, dataset='DFDC', frames=100, video_path="notebook/samples/mqzvfufzoq.mp4"):
    """
    Handle both single model and ensemble predictions
    
    Model parameter can be either:
    - A string for single model: 'EfficientNetB4', 'EfficientNetB4ST', etc.
    - A dict for ensemble: {
        'type': 'ensemble',
        'models': ['EfficientNetB4', 'Xception', ...],
        'weights': [0.4, 0.6, ...],
        'weight_paths': ['path1', 'path2', ...]
    }
    """
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    face_policy = 'scale'
    face_size = 224
    frames_per_video = frames

    # Initialize face detection components
    facedet = BlazeFace().to(device)
    facedet.load_weights("blazeface/blazeface.pth")
    facedet.load_anchors("blazeface/anchors.npy")
    videoreader = VideoReader(verbose=False)
    video_read_fn = lambda x: videoreader.read_frames(x, num_frames=frames_per_video)
    face_extractor = FaceExtractor(video_read_fn=video_read_fn, facedet=facedet)

    # Handle ensemble case
    if isinstance(model, dict) and model['type'] == 'ensemble':
        nets = []
        for model_name, weight_path in zip(model['models'], model['weight_paths']):
            net = getattr(fornet, model_name)().eval().to(device)
            state_dict = torch.load(weight_path, map_location=device)
            
            # Handle VGG special case
            if model_name == 'VGG16':
                new_state_dict = {}
                for key, value in state_dict.items():
                    if not key.startswith('vgg.'):
                        new_key = f'vgg.{key}'
                        new_state_dict[new_key] = value
                    else:
                        new_state_dict[key] = value
                net.load_state_dict(new_state_dict)
            else:
                net.load_state_dict(state_dict)
            nets.append(net)
        
        # Use first model's normalizer for transformation
        transf = utils.get_transformer(face_policy, face_size, nets[0].get_normalizer(), train=False)
    else:
        # Single model case
        net_model = model if isinstance(model, str) else model['model']
        model_path = os.path.join('weights', f'{net_model}_{dataset}_bestval.pth')
        net = getattr(fornet, net_model)().eval().to(device)
        
        state_dict = torch.load(model_path, map_location=device)
        if net_model == 'VGG16':
            new_state_dict = {}
            for key, value in state_dict.items():
                if not key.startswith('vgg.'):
                    new_key = f'vgg.{key}'
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            net.load_state_dict(new_state_dict)
        else:
            net.load_state_dict(state_dict)
        
        transf = utils.get_transformer(face_policy, face_size, net.get_normalizer(), train=False)

    # Process video and extract faces
    vid_fake_faces = face_extractor.process_video(video_path)
    faces_fake_t = torch.stack([transf(image=frame['faces'][0])['image'] for frame in vid_fake_faces if len(frame['faces'])])

    # Make predictions
    with torch.no_grad():
        if isinstance(model, dict) and model['type'] == 'ensemble':
            # Ensemble prediction
            predictions = []
            for i, net in enumerate(nets):
                pred = net(faces_fake_t.to(device)).cpu().numpy().flatten()[:50]
                predictions.append(pred * model['weights'][i])
            faces_fake_pred = sum(predictions)
        else:
            # Single model prediction
            faces_fake_pred = net(faces_fake_t.to(device)).cpu().numpy().flatten()

    # Calculate final prediction
    final_pred = expit(faces_fake_pred.mean())
    
    # Print debug information
    print(expit(faces_fake_pred))
    print(faces_fake_pred)
    print(final_pred)
    
    # Return result
    if final_pred > threshold:
        return 'fake', final_pred
    else:
        return 'real', final_pred