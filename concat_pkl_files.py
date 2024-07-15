import pickle

path1 = 'pretrained_vectors/emma_train_clips_mood_valence_tl100_tstep24_tstride3_part1.pkl'
with open(path1, 'rb') as f:
    data1 = pickle.load(f)
    print(f'Train data: {len(data1)}')


path1 = 'pretrained_vectors/emma_train_clips_mood_valence_tl100_tstep24_tstride3_part2.pkl'
with open(path1, 'rb') as f:
    data2 = pickle.load(f)
    print(f'Train data: {len(data2)}')


path1 = 'pretrained_vectors/emma_train_clips_mood_valence_tl100_tstep24_tstride3_part3.pkl'
with open(path1, 'rb') as f:
    data3 = pickle.load(f)
    print(f'Train data: {len(data3)}')

data1.extend(data2)
data1.extend(data3)

print(f'Train data: {len(data1)}')

with open('pretrained_vectors/emma_train_clips_mood_valence_tl100_tstep24_tstride3.pkl', 'wb') as f:
    pickle.dump(data1, f)
