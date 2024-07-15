import os
from PIL import Image
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from video_augmentation import AffWildAugmentor1, AffWildAugmentor2, AffWildAugmentor3


torch.manual_seed(5)

# Use this dataloader when there is a dataframe with the clip paths, mood annotation for the clip, delta similarity
# and delta diff. Getitem takes this dataframe as input.
# delta_similarity is obtained using the pre-trained siamese network SiameseEmoNetMLPMultiTask. The unique labels are
# 1 and 0, 1 for similar pair and 0 for dissimilar pair of frames.
# delta_diff is obtained as the sign of the difference in valence between first frame and the last frame in the clip.
# The unique delta_diff labels are 0, +1, and -1.
# mood annotation is derived using the valence labels. Unique mood labels are 0, +1, and -1.

# This dataloader gives clips of a video as batches. That is, one batch is all clips of a particular video. Use this
# when data has to be loaded at the video-level. The indices corresponding to the videos are created as a list and
# getitem iterates over this list.


class MoodEmotion(Dataset):
    # Characterizes a dataset
    def __init__(self, data_root, clip_df, batch_size=128, clip_height=64, clip_width=64, img_height=256, img_width=256,
                 augment=False):
        # Initialisation
        self.data_root = data_root
        # self.cfg = cfg
        # self.logger = logger
        self.clip_height = clip_height
        self.clip_width = clip_width
        self.img_height = img_height
        self.img_width = img_width
        self.clip_df = clip_df
        self.augment = augment
        self.batch_size = batch_size

        videos = list(self.clip_df['video_id'].unique())
        # print(len(videos))

        split = lambda lst, sz: [lst[i:i + sz] for i in range(0, len(lst), sz)]

        batches = []
        initial_index = 0
        for i in range(len(videos)):
            print(videos[i])
            sub_df = self.clip_df[self.clip_df['video_id'] == videos[i]]
            print(sub_df.shape[0])
            splits = split(np.arange(sub_df.shape[0]), self.batch_size)
            mini_batch = [len(i) for i in splits]
            # print(f' {len(sub_df)} : {mini_batch}')
            batches.append(mini_batch)
            # print(f'Batch indices: {batches}')
            # break

        batch_indices = [item for sublist in batches for item in sublist]
        batch_indices.insert(0, 0)
        self.batch_indices = list(np.cumsum(batch_indices))  # Final list with the batch indices.
        # print(batch_indices)

        self.transform_clip1 = transforms.Compose([
            transforms.Resize((self.clip_height, self.clip_width)),
            transforms.ToTensor()
        ])

        self.transform_img1 = transforms.Compose([
            transforms.Resize((self.img_height, self.img_width)),
            transforms.ToTensor()
        ])

        self.transform_img2 = transforms.Compose([
            transforms.Resize((self.img_height, self.img_width)),
            transforms.RandomAffine(degrees=(20, 45), translate=(0.1, 0.3), scale=(0.8, 1.2), shear=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.ColorJitter(brightness=.5, hue=.3),
            transforms.RandomCrop((self.img_height, self.img_width)),
            transforms.ToTensor(),
        ])

        self.transform_clip2 = transforms.Compose([
            transforms.Resize((self.clip_height, self.clip_width)),
            transforms.RandomAffine(degrees=(20, 45), translate=(0.1, 0.3), scale=(0.8, 1.2), shear=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.ColorJitter(brightness=.5, hue=.3),
            transforms.RandomCrop((self.clip_height, self.clip_width)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        # Denotes the total number of samples
        return len(self.batch_indices) - 1

    def __getitem__(self, index):
        # print("Index:", index)
        start_idx = self.batch_indices[index]
        # print("Start index:", start_idx)
        end_idx = self.batch_indices[index + 1]
        # print("End index:", end_idx)
        sub_df = self.clip_df.iloc[start_idx:end_idx]
        video_id = sub_df.iloc[0]['video_id']

        mood = self.clip_df.iloc[index]['mood']
        val1 = self.clip_df.iloc[index]['valence1']
        # print("Valence 1:", val1)
        val2 = self.clip_df.iloc[index]['valence2']
        # print("Valence 2:", val2)
        gt_delta_diff = self.clip_df.iloc[index]['gt_deltaval_diff']
        # print("GT Delta Diff:", gt_delta_diff)
        current_img = self.clip_df.iloc[index]['frame']
        current_val = self.clip_df.iloc[index]['frame_valence']

        if mood == 1.0:
            mood = int(1)
        elif mood == -1.0000:
            mood = int(2)
        elif mood == 0.0:
            mood = int(0)
        mood = torch.tensor(mood, dtype=torch.long)

        frames_batch = []
        for i in range(len(sub_df)):
            clip_path = eval(sub_df.iloc[i]['clip'])
            frames_list = []
            for j in clip_path:
                frame = Image.open(j).convert("RGB")
                frames_list.append(frame)
                if self.augment:
                    frames_list1 = [self.transform_clip2(frame) for frame in frames_list]
                else:
                    frames_list1 = [self.transform_clip1(frame) for frame in frames_list]
            frames = torch.stack(frames_list1, dim=0)
            # print(frames.shape)
            frames_batch.append(frames)
        frames_batch = torch.stack(frames_batch, dim=0)
        print(f'{video_id} : {sub_df.shape[0]}; {frames_batch.shape}')

        current_img = Image.open(current_img).convert("RGB")
        current_img = self.transform_img1(current_img)
        # print(current_img.shape)

        if 0.0 < gt_delta_diff <= 2.0:
            delta_val = int(1)
        elif 0.0 > gt_delta_diff >= -2.0:
            delta_val = int(2)
        else:
            delta_val = int(0)
        delta_val = torch.tensor(delta_val, dtype=torch.long)

        return dict(frames=frames_batch, video_id=video_id, mood=mood, valence1=val1, valence2=val2, gt_deltaval=gt_delta_diff,
                    current_image=current_img, current_valence=current_val, delta_val=delta_val)


# This dataloader uses video-level data augmentation for clips. It is used to load a clip and the frame immediately
# after the clip. The clip has a mood label and a delta label, whereas the frame has a valence label alone.


class MoodEmo(Dataset):
    # Characterizes a dataset
    def __init__(self, data_root, clip_df, cfg, clip_height=64, clip_width=64, img_height=256, img_width=256, augment=False):
        # Initialisation
        self.data_root = data_root
        self.cfg = cfg
        # self.logger = logger
        self.clip_height = clip_height
        self.clip_width = clip_width
        self.img_height = img_height
        self.img_width = img_width
        self.clip_df = clip_df
        self.augment = augment

        self.data_augmentation1 = AffWildAugmentor1(width=self.clip_width, height=self.clip_height)
        self.data_augmentation2 = AffWildAugmentor2(width=self.clip_width, height=self.clip_height)
        self.data_augmentation3 = AffWildAugmentor3(width=self.clip_width, height=self.clip_height)

        self.transform_clip1 = transforms.Compose([
            transforms.Resize((self.clip_height, self.clip_width)),
            transforms.ToTensor()
        ])

        self.transform_img1 = transforms.Compose([
            transforms.Resize((self.img_height, self.img_width)),
            transforms.ToTensor()
        ])

        self.transform_img2 = transforms.Compose([
            transforms.Resize((self.img_height, self.img_width)),
            transforms.RandomAffine(degrees=(20, 45), translate=(0.1, 0.3), scale=(0.8, 1.2), shear=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=.5, hue=.3),
            transforms.RandomCrop((self.img_height, self.img_width)),
            transforms.ToTensor(),
        ])

        self.transform_clip2 = transforms.Compose([
            transforms.Resize((self.clip_height, self.clip_width)),
            transforms.RandomAffine(degrees=(20, 45), translate=(0.1, 0.3), scale=(0.8, 1.2), shear=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=.5, hue=.3),
            transforms.RandomCrop((self.clip_height, self.clip_width)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        # Denotes the total number of samples
        return len(self.clip_df)

    def __getitem__(self, index):
        video_id = self.clip_df.iloc[index]['video_id']
        # print("Video ID:", video_id)
        clip_path = eval(self.clip_df.iloc[index]['clip'])
        mood = self.clip_df.iloc[index]['mood']
        # print("Mood:", mood)
        val1 = self.clip_df.iloc[index]['valence1']
        # print("Valence 1:", val1)
        val2 = self.clip_df.iloc[index]['valence2']
        # print("Valence 2:", val2)
        gt_delta_diff = self.clip_df.iloc[index]['gt_deltaval_diff']
        # print("GT Delta Diff:", gt_delta_diff)
        current_img = self.clip_df.iloc[index]['frame']
        current_val = self.clip_df.iloc[index]['frame_valence']
        # print("Current Valence:", current_val)

        frames_list1 = []
        # print(frames_list[0])
        # print(len(frames_list))
        for i in clip_path:
            # print(i)
            frame = Image.open(i).convert("RGB")
            frames_list1.append(frame)

        if self.augment:
            clip = self.data_augmentation1(frames_list1)
            frames_list2 = [self.transform_clip1(frame) for frame in clip]
        else:
            clip = self.data_augmentation3(frames_list1)
            frames_list2 = [self.transform_clip1(frame) for frame in clip]

        frames = torch.stack(frames_list2, dim=1)

        if mood == 1.0:
            mood = int(1)
        elif mood == -1.0000:
            mood = int(2)
        elif mood == 0.0:
            mood = int(0)
        mood = torch.tensor(mood, dtype=torch.long)
        # print("Mood:", mood)

        # img1 = Image.open(clip_path[0]).convert("RGB")
        # img1 = self.transform(img1)
        # print(img1.shape)
        # img2 = Image.open(clip_path[-1]).convert("RGB")
        # img2 = self.transform(img2)
        img1 = frames[:, 0, :, :]
        img2 = frames[:, -1, :, :]
        # print(img2.shape)

        current_img = Image.open(current_img).convert("RGB")
        current_img = self.transform_img1(current_img)
        # print(current_img.shape)

        if 0.0 < gt_delta_diff <= 2.0:
            delta_val = int(1)
        elif 0.0 > gt_delta_diff >= -2.0:
            delta_val = int(2)
        else:
            delta_val = int(0)
        delta_val = torch.tensor(delta_val, dtype=torch.long)

        return dict(video_id=video_id, frames=frames, mood=mood, valence1=val1, valence2=val2, gt_deltaval=gt_delta_diff,
                    image1=img1, image2=img2, current_image=current_img, current_valence=current_val, delta_val=delta_val)


def visualise_emmaclip(dataset, num_frames, batch_size=1):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    data_dict = next(iter(dataloader))

    # plot data
    # To have dark background
    # plt.style.use('dark_background')
    fig, axes = plt.subplots(1, num_frames, figsize=(10, 5))

    # Visualise first sample in the batch
    for i in range(num_frames):
        clip1 = data_dict["frames"][0, :, i, :, :]
        # print(clip1.shape)
        clip1 = clip1.permute(1, 2, 0)
        # print(clip1.shape)
        axes[i].imshow(clip1)
        # axes[i].set_title(data_dict['frames'][i][-9:])
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    from tqdm import tqdm
    from collections import Counter
    from pathlib import Path
    import datetime
    import pickle
    import pandas as pd

    path_to_csv1 = "emma_video_frames.csv"
    data_root = "../EMMA/EMMA_all_faces_frames"
    video_df = pd.read_csv(path_to_csv1)
    # print(video_df)
    path_to_csv2 = "emma_train_clips_mood_valence_tl100_tstep24_tstride3.csv"
    clip_df = pd.read_csv(path_to_csv2)
    # print(clip_df)
    # train_pkl = "pretrained_vectors/emma_train_clips_mood_valence_tl100_tstep24_tstride3.pkl"
    # with open(train_pkl, 'rb') as f:
    #     data = pickle.load(f)

    dataset = EmmaMoodEmo(data_root, clip_df, height=64, width=64, augment=False)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=2)
    delta_sim = []
    for tr_batch_idx, train_dict in enumerate(tqdm(dataloader)):
        print(tr_batch_idx)
        print("Final size:", train_dict['frames'].shape)
        # print(train_dict['video_id'])
        # print(train_dict['current_clip_path'])
        # print(train_dict['previous_clip_path'])
        # print(train_dict['current_img_path'])
        # print(train_dict['previous_img_path'])
        # break
        if tr_batch_idx == 3:
            break

    # print(y_true_dict)
    visualise_emmaclip(dataset, num_frames=5, batch_size=10)



