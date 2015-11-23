from pymongo import MongoClient
import pickle
import Utils as util
import Consts as c
import dao
# perfect match is a match which all the players are in my player dictionary


player_dict_path = 'data/player_dict.pickle'
mid_path_test = 'data/perfect_match_ids_test.pickle'
mid_path = 'data/perfect_match_ids.pickle'
perfect_match_path = 'data/perfect_matches.pickle'
perfect_match_path_test = 'data/perfect_matches_test.pickle'
match_limit = 5000
match_per_player = 300


def find_perfect_matches(save_path):
    # Load player dict
    p_dict = util.load_pickle_file(player_dict_path)
    p_ids = p_dict.keys()
    # load matches
    match_col = dao.get_match_col()

    print('Start searching...')
    perfect_matches_ids = []
    done = False

    for m in match_col.find():
        m_id = m[c.MATCH_ID]
        if m_id not in perfect_matches_ids and match_col.find({c.MATCH_ID : m_id}).count() == 10:
            is_perfect = True
            for mm in match_col.find({c.MATCH_ID : m_id}):
                p_id = mm[c.SUMMONER_ID]
                if match_col.find({c.SUMMONER_ID : p_id}).count() < match_per_player:
                    is_perfect = False
                    break
            if is_perfect:
                perfect_matches_ids.append(m_id)
                print('Matches found: {}'.format(len(perfect_matches_ids)))
                if len(perfect_matches_ids) >= match_limit:
                    done = True
                    break

    print('Saving results...')
    util.save(perfect_matches_ids, save_path)
    print('Find {} perfect matches!'.format(len(perfect_matches_ids)))


def find_perfect_matches_bak(save_path):
    # Load player dict
    p_dict = util.load_pickle_file(player_dict_path)
    p_ids = p_dict.keys()
    # TODO load matches
    client = MongoClient('mongodb://foo:bar@nb4800.neu.edu:12857/lol?authMechanism=SCRAM-SHA-1')
    lol_db = client['lol']
    match_col = dao.get_match_col()

    print('Start searching...')
    perfect_matches_ids = []
    done = False
    for id in p_ids:
        for m in match_col.find({c.SUMMONER_ID : id}):
            mid = m[c.MATCH_ID]
            if mid not in perfect_matches_ids:
                is_perfect_match = True
                player_count = 0
                for mm in match_col.find({c.MATCH_ID : mid}):
                    if mm[c.SUMMONER_ID] not in p_ids:
                        is_perfect_match = False
                        break
                    player_count += 1
                if is_perfect_match and player_count == 10:
                    perfect_matches_ids.append(mid)
                    print('Matches found: {}'.format(len(perfect_matches_ids)))
                    if len(perfect_matches_ids) >= match_limit:
                        done = True
                        break
        if done:
            break


    print('Saving results...')
    util.save(perfect_matches_ids, save_path)
    print('Find {} perfect matches!'.format(len(perfect_matches_ids)))



def get_perfect_match_data(id_path, save_path):
    mid_list = util.load_pickle_file(id_path)
    match_col = dao.get_match_col()

    perfect_matches = {}  # teams: player ids, gold, win?

    for mid in mid_list:
        win_team = {}
        lose_team = {}
        perfect_matches[mid] = [win_team, lose_team]
        init_team(win_team)
        init_team(lose_team)
        for m in match_col.find({c.MATCH_ID : mid}):
            if m[c.WINNER]:
                win_team['players'].append(m[c.SUMMONER_ID])
                win_team['goldEarned'] += m[c.GOLD_EARNED]
                win_team['goldSpent'] += m[c.GOLD_SPENT]
            else:
                lose_team['players'].append(m[c.SUMMONER_ID])
                lose_team['goldEarned'] += m[c.GOLD_EARNED]
                lose_team['goldSpent'] += m[c.GOLD_SPENT]
    util.save(perfect_matches, save_path)
    print(perfect_matches[mid_list[0]])

def init_team(t):
    t['players'] = []
    t['goldEarned'] = 0
    t['goldSpent'] = 0


if __name__ == '__main__':
    find_perfect_matches(mid_path)
    # get_perfect_match_data(mid_path_test, perfect_match_path_test)
    # get_perfect_match_data(mid_path, perfect_match_path)