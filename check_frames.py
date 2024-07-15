import pandas as pd
import numpy as np
from tqdm import tqdm
import os

df = pd.read_csv('emma_clips_mood_valence_tl100_tstep24_tstride3.csv')
# df = pd.read_csv('emma_valence_frame_level_reduced.csv')

videos = list(df['video_id'].unique())
videos.remove('0422a')
videos.remove('0422b')
videos.remove('1025a')
videos.remove('1221')

for i in tqdm(range(len(videos))):
    video = videos[i]
    # print(video)
    sub_df = df[df['video_id'] == video]
    # print(len(sub_df))
    folder = os.path.join('../EMMA/EMMA_all_faces_frames', video)
    print(f'{video}: Clips: {len(sub_df)}')
    # break




