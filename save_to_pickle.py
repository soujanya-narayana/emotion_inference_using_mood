import pandas as pd
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torch
from tqdm import tqdm
import numpy as np
import pickle
from mood_model import generate_model_branchedlinear
from delta_model import SiameseNetEmoNetMLP
from emonet import EmoNet
from utils import load_config
from pathlib import Path

cfg = load_config(str(Path(__file__).parent.joinpath('config.yaml')))

df = pd.read_csv('emma_test_clips_mood_valence_tl100_tstep24_tstride3.csv')


def preprocess_clip(clip_path):
    clip_path = eval(clip_path)
    transform = transforms.Compose([
        transforms.Resize((cfg['IMAGE_SHAPE'], cfg['IMAGE_SHAPE'])),
        transforms.ToTensor()
    ])

    frames = torch.empty((3, 5, cfg['IMAGE_SHAPE'], cfg['IMAGE_SHAPE']))
    for i in range(cfg['NUM_FRAMES']):
        frame = Image.open(clip_path[i]).convert("RGB")
        frame = transform(frame)
        frames[:, i, :, :] = frame

    img1 = frames[:, 0, :, :]
    img2 = frames[:, -1, :, :]

    return frames, img1, img2


def preprocess_img(img_path):
    transform = transforms.Compose([
        transforms.Resize((cfg['IMAGE_SHAPE'], cfg['IMAGE_SHAPE'])),
        transforms.ToTensor()
    ])

    img = Image.open(img_path).convert("RGB")
    img = transform(img)

    return img


mood_model = generate_model_branchedlinear(model_depth=cfg["RESNET_DEPTH"], n_classes_b1=cfg["CLASS_NUM_BRANCH1"],
                                           n_classes_b2=cfg["CLASS_NUM_BRANCH2"], dropout_rate=cfg["DROPOUT_RATE"]).to(cfg["DEVICE"])
mood_model.eval()

delta_model = SiameseNetEmoNetMLP(emofan_n_emotions=cfg["EMOFAN_EMOTIONS"], is_pretrained_emofan=cfg["IS_PRETRAINED_EMOFAN"],
                                  feat_fusion=cfg["FEAT_FUSION_MTCLAR"], num_neurons=cfg["NUM_NEURONS_MTCLAR"],
                                  dropout_rate=cfg["DROPOUT_RATE"], is_multi_task=True).to(cfg["DEVICE"])
delta_model.eval()

# Pretrained EmoFAN model
emofan = EmoNet(num_modules=2, n_expression=cfg["EMOFAN_EMOTIONS"], n_reg=2, n_blocks=4, attention=True,
                temporal_smoothing=False).to(cfg["DEVICE"])
emofan.eval()

# Load weights to mood model
state_dict_path1 = Path(__file__).parent.joinpath('pretrained', f'mood_best_model.pth')
print(f'Loading mood model from {state_dict_path1}...')
state_dict1 = torch.load(str(state_dict_path1), map_location=torch.device('cuda:0'))
state_dict1 = {k.replace('module.', ''): v for k, v in state_dict1.items()}
mood_model.load_state_dict(state_dict1, strict=False)

# Load weights to delta model
state_dict_path2 = Path(__file__).parent.joinpath('pretrained', f'delta_best_model.pth')
print(f'Loading delta model from {state_dict_path2}...')
state_dict2 = torch.load(str(state_dict_path2), map_location=torch.device('cuda:0'))
state_dict2 = {k.replace('module.', ''): v for k, v in state_dict2.items()}
delta_model.load_state_dict(state_dict2, strict=False)


state_dict_path3 = Path(__file__).parent.joinpath('pretrained', f'emonet_8.pth')
print(f'Loading emofan model from {state_dict_path3}...')
state_dict3 = torch.load(str(state_dict_path3), map_location=torch.device('cuda:0'))
state_dict3 = {k.replace('module.', ''): v for k, v in state_dict3.items()}
emofan.load_state_dict(state_dict3, strict=False)

fc_in_features = emofan.emo_fc_2[0].in_features  # [bs, 256]
mood_features = mood_model.fc_b1_1[6]  # [bs, 256]
delta_features = delta_model.fc0_1[11]  # [bs, 512]

# get features before the last linear block
# Refer emonet supplementary material for more info
emofan.emo_fc_2 = nn.Identity()

data = []
for index in tqdm(range(df.shape[0])):
    clip_path = df.iloc[index, :]['clip']
    # print('Clip path', clip_path)
    mood_annotation = df.iloc[index, :]['mood']
    frame_path = df.iloc[index, :]['frame']
    frame_valence = df.iloc[index, :]['frame_valence']
    # print('Frame val', frame_valence)

    # Load and preprocess the frame
    clip, img1, img2 = preprocess_clip(clip_path)
    img = preprocess_img(frame_path)

    img1_valence = df.iloc[index, :]['valence1']
    img2_valence = df.iloc[index, :]['valence2']
    gt_delta_valence = df.iloc[index, :]['gt_deltaval_diff']

    # Pass the frame through the pre-trained model to obtain the vector
    with torch.no_grad():
        mood_output = mood_model(clip.unsqueeze(0).to(cfg["DEVICE"]))  # Assuming your model expects batch dimension
        mood_vector = mood_output['feat1'].view(mood_output['feat1'].size()[0], -1)
        # print('Mood vector', mood_vector.shape)

        delta_output = delta_model(img1.unsqueeze(0).to(cfg["DEVICE"]), img2.unsqueeze(0).to(cfg["DEVICE"]))
        delta_vector = delta_output['feat0'].view(delta_output['feat0'].size()[0], -1)
        # print('Delta vector', delta_vector.shape)

        emofan_output = emofan(img.unsqueeze(0).to(cfg["DEVICE"]))
        emofan_vector = emofan_output['feature'].view(emofan_output['feature'].size()[0], -1)
        # print('Emofan vector', emofan_vector.shape)

    # Create a dictionary to store the data
    clip_info = {
        'clip_path': clip_path,
        'mood': mood_annotation,
        'valence1': img1_valence,
        'valence2': img2_valence,
        'frame_valence': frame_valence,
        'gt_delta_valence': gt_delta_valence,
        'mood_vector': mood_vector,
        'delta_vector': delta_vector,
        'image_vector': emofan_vector
    }

    data.append(clip_info)
    #if index == 10:
    #    break
    # break

# Save the clip data as a pickle file
with open('emma_test_clips_mood_valence_tl100_tstep24_tstride3.pkl', 'wb') as f:
    pickle.dump(data, f)


with open('emma_test_clips_mood_valence_tl100_tstep24_tstride3.pkl', 'rb') as f:
    data_new = pickle.load(f)
    print('Data new', len(data_new))




