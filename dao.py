from pymongo import MongoClient



def get_match_col():
    client = MongoClient('mongodb://foo:bar@nb4800.neu.edu:12857/lol?authMechanism=SCRAM-SHA-1')
    lol_db = client['lol']
    match_col = lol_db['match']
    return match_col
