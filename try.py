import pickle

player_dict_path = 'data/player_dict.pickle'
with open(player_dict_path, 'rb') as f:
    p_dict = pickle.load(f)

print('done')
