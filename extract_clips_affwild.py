import random

import pandas as pd
import numpy as np
import os
from tqdm import tqdm


df = pd.read_csv('emma_test_individual_frames.csv')
# print(df)
temporal_length = 200  # refers to the length of the clip in terms of number of frames
temporal_step = 49  # step value between two consecutive frames in a clip
temporal_stride = 3  # sliding window stride value for extracting clips from a video
num_frames = 5  # number of frames in a clip

# The final dataframe will have the following columns: clip; referring to a list of the paths of the frames in a clip,
# mood; referring to the mood of the video, frame; referring to the path of the frame after the clip,
# valence; referring to the valence of the frame after the clip,
# gt_deltaval_diff; referring to the difference between the valence of the last frame of the clip and the first frame of the clip.

videos = pd.unique(df['video_id'].to_list())
print(len(videos))

'''
final_df = pd.DataFrame(columns=['video_id', 'clip', 'mood', 'valence1', 'valence2', 'frame', 'frame_valence', 'gt_deltaval_diff'])
for video in tqdm(videos):
    # print("Video:", video)
    sub_df = df[df['video_id'] == video]  # dataframe for a particular video
    print(f'{video} : {len(sub_df)} Duplicates: {sub_df["frames"].duplicated().any()}')  # check if there are any duplicate frame numbers
    sub_df.drop_duplicates(subset=['frames'], inplace=True)  # remove duplicate frame numbers
    sub_df = sub_df.reset_index(drop=True)
    sub_dir = os.path.join('../AffWild/affwild_cropped/cropped_aligned', video)
    idx = []
    for j in range(1, len(sub_df) - temporal_length + 1, temporal_stride):
        sub_dict = {}
        clip = []
        for k in range(j, temporal_length + j, temporal_step):
            # print(k)
            frame_path = os.path.join(sub_dir, str(k).zfill(5) + '.jpg')  # path of the frame
            clip.append(frame_path)  # append paths of 5 frames to the clip
        sub_dict['video_id'] = video
        sub_dict['clip'] = [clip]
        sub_dict['mood'] = sub_df['mood'].unique()[0]  # mood of the video
        img1 = clip[0][-9:]
        sub_dict['valence1'] = sub_df.loc[sub_df['frames'] == img1, 'valence'].values[0]
        img2 = clip[-1][-9:]
        sub_dict['valence2'] = sub_df.loc[sub_df['frames'] == img2, 'valence'].values[0]
        current_img = str(int(img2[:-4].lstrip("0")) + 1).zfill(5) + '.jpg'
        img_path = os.path.join(sub_dir, current_img)  # path of the next frame after the clip

        sub_dict['frame'] = img_path
        frame = int(img2[:-4].lstrip("0")) + 1
        sub_dict['frame_valence'] = sub_df.loc[sub_df['frames'] == current_img, 'valence'].values[0]  # valence of the next frame after the clip
        sub_dict['gt_deltaval_diff'] = sub_df.loc[sub_df['frames'] == img2, 'valence'].values[0] -\
                                       sub_df.loc[sub_df['frames'] == img1, 'valence'].values[0]
        new_df = pd.DataFrame(sub_dict)
        final_df = pd.concat([final_df, new_df], ignore_index=True)
        # print(new_df)
    # break

print(final_df)
final_df.to_csv('affwild_clips_test_mood_valence_7frames_tl250_tstep41_tstride3.csv', index=False)
'''

final_df = pd.DataFrame(columns=['video_id', 'clip', 'mood', 'valence1', 'valence2', 'frame', 'frame_valence', 'gt_deltaval_diff'])
for video in tqdm(videos):
    # print("Video:", video)
    sub_df = df[df['video_id'] == video]  # dataframe for a particular video
    print(f'{video} : {len(sub_df)} Duplicates: {sub_df["frame_number"].duplicated().any()}')  # check if there are any duplicate frame numbers
    sub_df.drop_duplicates(subset=['frame_number'], inplace=True)  # remove duplicate frame numbers
    sub_df = sub_df.reset_index(drop=True)
    sub_dir = os.path.join('../AffWild/affwild_cropped/cropped_aligned', video)
    idx = []
    for j in range(1, len(sub_df) - temporal_length + 1, temporal_stride):
        sub_dict = {}
        clip = []
        for k in range(j, temporal_length + j, temporal_step):
            # print(k)
            frame_path = os.path.join(sub_dir, str(k).zfill(5) + '.jpg')  # path of the frame
            clip.append(frame_path)  # append paths of 5 frames to the clip
        sub_dict['video_id'] = video
        sub_dict['clip'] = [clip]
        sub_dict['mood'] = sub_df['mood'].unique()[0]  # mood of the video
        img1 = clip[0][-9:][:-4].lstrip("0")  # first frame of the clip
        # print('Image1:', img1)
        sub_dict['valence1'] = sub_df.loc[sub_df['frame_number'] == int(img1), 'valence'].values[0]  # valence of the first frame of the clip
        img2 = clip[-1][-9:][:-4].lstrip("0")  # last frame of the clip
        sub_dict['valence2'] = sub_df.loc[sub_df['frame_number'] == int(img2), 'valence'].values[0]  # valence of the last frame of the clip
        # print(clip[-1])
        # print('Image2:', img2)
        img_path = os.path.join(sub_dir, str(int(img2) + 1).zfill(5) + '.jpg')  # path of the next frame after the clip
        sub_dict['frame'] = img_path
        frame = int(img2[:-4].lstrip("0")) + 1
        sub_dict['frame_valence'] = sub_df.loc[sub_df['frame_number'] == frame, 'valence'].values[0]  # valence of the next frame after the clip
        sub_dict['gt_deltaval_diff'] = sub_df.loc[sub_df['frame_number'] == int(img2), 'valence'].values[0] -\
                                       sub_df.loc[sub_df['frame_number'] == int(img1), 'valence'].values[0]
        new_df = pd.DataFrame(sub_dict)
        final_df = pd.concat([final_df, new_df], ignore_index=True)
        print(new_df)
        break
    break

# print(final_df)
# final_df.to_csv('emma_clips_mood_valence_5frames_tl100_tstep24_tstride5.csv', index=False)

print(final_df)
final_df.to_csv("emma_long_clips_test_mood_valence_5frames_tl200_tstep49_tstride3.csv", index=False)

