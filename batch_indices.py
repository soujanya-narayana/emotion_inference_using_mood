import pandas as pd
import numpy as np

df = pd.read_csv('emma_clips_mood_valence_tl100_tstep24_tstride3.csv')

videos = list(df['video_id'].unique())
# print(len(videos))

split = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]
batch_size = 128

total_samples = []
batches = []
initial_index = 0
for i in range(len(videos)):
    sub_df = df[df['video_id'] == videos[i]]
    total_samples.append(len(sub_df))
    splits = split(np.arange(len(sub_df)), batch_size)
    mini_batch = [len(i) for i in splits]
    # print(f' {len(sub_df)} : {mini_batch}')
    batches.append(mini_batch)
    # print(f'Batch indices: {batches}')
    # break

batch_indices = [item for sublist in batches for item in sublist]
# print(batch_indices)
batch_indices.insert(0, 0)
batch_indices = list(np.cumsum(batch_indices))
print(batch_indices)



