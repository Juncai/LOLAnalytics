from pymongo import MongoClient


client = MongoClient('mongodb://Admin:Plait2014Lab@blazing.ccs.neu.edu:27017/admin')
vp_db = client['videotracker_Production']
lz_col = vp_db['VideoValence']
print(lz_col.count())


match_dict = {}
# find all match id with 10 summoners
# all_ids = match_col.distinct('match_id')  # exceeds 16mb cap
p = [{"$group": {"_id": "$valence"}}, {'$limit': 5}]
some_ids = list(lz_col.aggregate(p))
print(some_ids)