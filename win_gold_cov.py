# from pymongo import MongoClient
import pickle
import numpy as np

# TODO load matches
# winner bool
# goldEarned int32
# goldSpent int32
# client = MongoClient('mongodb://foo:bar@localhost:27017/lol?authMechanism=SCRAM-SHA-1')
# lol_db = client['lol']
# match_col = lol_db['match']
# print(match_col.count())


# find all match id with 10 summoners
# all_ids = match_col.distinct('match_id')  # exceeds 16mb cap
# p = [{"$group": {"_id": "$match_id"}}, {'$limit': 100000}]
# some_ids = list(match_col.aggregate(p))
# # print(some_ids)

# TODO find 10000 match with 10 players data
path = 'data/match_ids'
# match_id_set = set()
# for m in match_col.find():
#     mid = m["match_id"]
#     if mid not in match_id_set and match_col.find({"match_id" : mid}).count() == 10:
#         match_id_set.add(mid)
#         print(len(match_id_set))
#     if len(match_id_set) == 10000:
#         break
#
# with open(path, 'wb+') as f:
#     pickle.dump(match_id_set, f)


dict_path = 'data/match_dict'
# with open(path, 'rb') as f:
#     match_id_set = pickle.load(f)
# match_dict = {}
# cnt = 1
# for mid in match_id_set:
#     gold_earned_winner = 0
#     gold_spent_winner = 0
#     gold_earned_loser = 0
#     gold_spent_loser = 0
#
#     for m in match_col.find({"match_id" : mid}):
#         if m["winner"]:
#             print("winner, cnt: {}".format(cnt))
#             gold_earned_winner += m["goldEarned"]
#             gold_spent_winner += m["goldSpent"]
#         else:
#             print("loser, cnt: {}".format(cnt))
#             gold_earned_loser += m["goldEarned"]
#             gold_spent_loser += m["goldSpent"]
#     match_dict[mid] = [gold_earned_winner, gold_spent_winner, gold_earned_loser, gold_spent_loser]
#     cnt += 1
# with open(dict_path, 'wb+') as f:
#     pickle.dump(match_dict, f)

with open(dict_path, 'rb') as f:
    match_dict = pickle.load(f)

# gold_earned_diffs = []
# gold_spent_diffs = []
# # wins = []
# wins_count = 0
# for k in match_dict.keys():
#     m = match_dict[k]
#     wins_count += 1 if m[0] > m[1] else 0
#     gold_earned_diffs.append(1 if m[0] > m[2] else 0)
#     gold_spent_diffs.append(1 if m[1] > m[3] else 0)
# n = len(gold_earned_diffs)
# wins = np.ones((1, n)).tolist()[0]
# print('gold_earned_win_cov: {}'.format(np.cov(gold_earned_diffs, wins)))
# print('gold_earned_win_cov: {}'.format(np.cov(gold_spent_diffs, wins)))
# tmp1 = np.dot(gold_earned_diffs, wins) / n
# tmp2 = np.dot(gold_spent_diffs, wins) / n
# print('gold_earned_acc: {}'.format(tmp1))
# print('gold_spent_acc: {}'.format(tmp2))

wins_count_earned = 0
wins_count_spent = 0
for k in match_dict.keys():
    m = match_dict[k]
    wins_count_earned += 1 if m[0] > m[1] else 0
    wins_count_spent += 1 if m[2] > m[3] else 0
print('p(win, more gold earned): {}'.format(wins_count_earned / len(match_dict)))
print('p(win, more gold spent): {}'.format(wins_count_spent / len(match_dict)))