import pandas as pd
import numpy as np
import os
from tqdm import tqdm


#df = pd.read_csv('affwild_train_individual_frames_valence_mood_expression.csv', dtype={'video': str})
df = pd.read_csv('emma_test_reduced_individual_frames.csv')
# df['video_id'] = [i.zfill(4) if len(i) == 3 else i for i in np.array(df['video_id'])]
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

final_df = pd.DataFrame(columns=['video_id', 'clip', 'mood', 'valence1', 'valence2', 'frame', 'frame_valence', 'gt_deltaval_diff'])
for video in tqdm(videos):
    # print("Video:", video)
    sub_df = df[df['video_id'] == video]  # dataframe for a particular video
    sub_dir = os.path.join('../EMMA/EMMA_test_faces_extracted', video)
    idx = []
    a = np.arange(sub_df.shape[0])
    for i in range(temporal_length, sub_df.shape[0], temporal_stride):
        sub_dict = {}
        clip = []
        c = [a[j] for j in range(1, i)]
        # print("c is", c)
        d = np.linspace(c[0], c[len(c) - 1], num_frames)   # returns num_frames evenly spaced frames from list c, starting from the first frame each time.
        # print("d:", d)
        d1 = [int(x) for x in d]
        for num in d1:
            frame_path = os.path.join(sub_dir, str(num).zfill(5) + '.jpg')  # path of the frame
            clip.append(frame_path)  # append paths of 5 frames to the list clip
        # print(clip)
        sub_dict['video_id'] = video
        sub_dict['clip'] = [clip]
        # print(sub_dict)
        sub_dict['mood'] = sub_df['mood'].unique()[0]  # mood of the video
        a1 = clip[0][-9:]
        a2 = a1.split('.')[0]
        img1 = int(a2.lstrip('0'))
        sub_dict['valence1'] = sub_df.loc[sub_df['frames'] == img1, 'valence'].values[0]
        a3 = clip[-1][-9:]
        a4 = a3.split('.')[0]
        img2 = int(a4.lstrip('0'))
        sub_dict['valence2'] = sub_df.loc[sub_df['frames'] == img2, 'valence'].values[0]
        current_img = str(int(img2) + int(1)).zfill(5) + '.jpg'
        # current_img = str(int(img2[:-4].lstrip("0")) + 1).zfill(5) + '.jpg'  # use for affwild
        img_path = os.path.join(sub_dir, current_img)  # path of the next frame after the clip
        sub_dict['frame'] = img_path
        a5 = current_img.split('.')[0]
        a6 = int(a5.lstrip('0'))
        sub_dict['frame_valence'] = sub_df.loc[sub_df['frames'] == a6, 'valence'].values[0]  # valence of the next frame after the clip
        sub_dict['gt_deltaval_diff'] = sub_df.loc[sub_df['frames'] == img2, 'valence'].values[0] - \
                                       sub_df.loc[sub_df['frames'] == img1, 'valence'].values[0]
        new_df = pd.DataFrame(sub_dict)
        # print(new_df)
        final_df = pd.concat([final_df, new_df], ignore_index=True)
        # break
    # break

print(final_df)
final_df.to_csv("test_emma_long_clips_test_mood_valence_5frames_tl200_tstep49_tstride3.csv", index=False)

'''
videos = pd.unique(df['video'].to_list())
# print("Total videos:", len(videos))

directory = '../EMMA/EMMA_test_faces'
files = os.listdir(directory)
# print(len(files))

test_df = pd.DataFrame()
for i in files:
    sub_df = df[df['video'] == i]
    test_df = pd.concat([test_df, sub_df])

print(test_df.shape)
test_df.to_csv("emma_test_individual_frames.csv", index=False)
'''


