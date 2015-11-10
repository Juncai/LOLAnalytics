from pymongo import MongoClient
import pickle
import numpy as np
import os.path as path


SUMMONER_ID = 'summoner_id'
ASSISTS = 'assists'
CHAMP_LEVEL = 'champLevel'
COMBAT_PLAYER_SCORE = 'combatPlayerScore'
DEATHS = 'deaths'
DOUBLE_KILLS = 'doubleKills'
FIRST_BLOOD_ASSIST = 'firstBloodAssist'
FIRST_BLOOD_KILL = 'firstBloodKill'
FIRST_INHIBITOR_ASSIST = 'firstInhibitorAssist'
FIRST_INHIBITOR_KILL = 'firstInhibitorKill'
FIRST_TOWER_ASSIST = 'firstTowerAssist'
FIRST_TOWER_KILL = 'firstTowerKill'
GOLD_EARNED = 'goldEarned'
GOLD_SPENT = 'goldSpent'
INHIBITOR_KILLS = 'inhibitorKills'
KILLING_SPREES = 'killingSprees'
KILLS = 'kills'
LARGEST_CRITICAL_STRIKE = 'largestCriticalStrike'
LARGEST_KILLING_SPREE = 'largestKillingSpree'
LARGEST_MULTI_KILL = 'largestMultiKill'
MAGIC_DAMAGE_DEALT = 'magicDamageDealt'
MAGIC_DAMAGE_DEALT_TO_CHAMPIONS = 'magicDamageDealtToChampions'
MAGIC_DAMAGE_TAKEN = 'magicDamageTaken'
MINIONS_KILLED = 'minionsKilled'
NEUTRAL_MINIONS_KILLED = 'neutralMinionsKilled'
NEUTRAL_MINIONS_KILLED_ENEMY_JUNGLE = 'neutralMinionsKilledEnemyJungle'
NEUTRAL_MINIONS_KILLED_TEAM_JUNGLE = 'neutralMinionsKilledTeamJungle'
PENTA_KILLS = 'pentaKills'
PHYSICAL_DAMAGE_DEALT = 'physicalDamageDealt'
PHYSICAL_DAMAGE_DEALT_TO_CHAMPIONS = 'physicalDamageDealtToChampions'
PHYSICAL_DAMAGE_TAKEN = 'physicalDamageTaken'
QUADRA_KILLS = 'quadraKills'
SIGHT_WARDS_BOUGHT_IN_GAME = 'sightWardsBoughtInGame'
TOTAL_DAMAGE_DEALT = 'totalDamageDealt'
TOTAL_DAMAGE_DEALT_TO_CHAMPIONS = 'totalDamageDealtToChampions'
TOTAL_DAMAGE_TAKEN = 'totalDamageTaken'
TOTAL_HEAL = 'totalHeal'
TOTAL_TIME_CROWD_CONTROL_DEALT = 'totalTimeCrowdControlDealt'
TOTAL_UNITS_HEALED = 'totalUnitsHealed'
TOWER_KILLS = 'towerKills'
TRIPLE_KILLS = 'tripleKills'
TRUE_DAMAGE_DEALT = 'trueDamageDealt'
TRUE_DAMAGE_DEALT_TO_CHAMPIONS = 'trueDamageDealtToChampions'
TRUE_DAMAGE_TAKEN = 'trueDamageTaken'
UNREAL_KILLS = 'unrealKills'
VISION_WARDS_BOUGHT_IN_GAME = 'visionWardsBoughtInGame'
WARDS_KILLED = 'wardsKilled'
WARDS_PLACED = 'wardsPlaced'

MATCH_ID = 'match_id'
MATCH_ID_LIST = 'matches'
MATCH_COUNT = 'match_num'
FEATURES = 'features'


def main():
    # TODO load matches
    client = MongoClient('mongodb://foo:bar@nb4800.neu.edu:12857/lol?authMechanism=SCRAM-SHA-1')
    lol_db = client['lol']
    match_col = lol_db['match']

    target_player_num = 1000
    matches_per_player = 500
    player_dict_path = 'data/player_dict.pickle'
    player_dict = {}
    player_count = 0

    print('Start retrieving...')
    for m in match_col.find():
        p_id = m[SUMMONER_ID]
        if p_id not in player_dict.keys():
            m_count = match_col.find({SUMMONER_ID : p_id}).count()
            if m_count > matches_per_player:
                p_m_count = 0
                for mm in match_col.find({SUMMONER_ID : p_id}):
                    if sanity_check(mm):
                        if p_m_count == 0:
                            init_player(player_dict, p_id, mm)
                        else:
                            update_player(player_dict, p_id, mm)
                        p_m_count += 1
                    if p_m_count == matches_per_player:
                        break
                if p_m_count < matches_per_player:
                    player_dict.pop(p_id)
                else:
                    player_count += 1
                    print('Finished players {} with player id {}'.format(player_count, p_id))
        if player_count == target_player_num:
            break

    print('Writing dict to file...')
    with open(player_dict_path, 'wb+') as f:
        pickle.dump(player_dict, f)

    print('Done!')
    return

def sanity_check(m):
    if ASSISTS not in m.keys(): return False
    if CHAMP_LEVEL not in m.keys(): return False
    if COMBAT_PLAYER_SCORE not in m.keys(): return False
    if DEATHS not in m.keys(): return False
    if DOUBLE_KILLS not in m.keys(): return False
    if FIRST_BLOOD_ASSIST not in m.keys(): return False
    if FIRST_BLOOD_KILL not in m.keys(): return False
    if FIRST_INHIBITOR_ASSIST not in m.keys(): return False
    if FIRST_INHIBITOR_KILL not in m.keys(): return False
    if FIRST_TOWER_ASSIST not in m.keys(): return False
    if FIRST_TOWER_KILL not in m.keys(): return False
    if GOLD_EARNED not in m.keys(): return False
    if GOLD_SPENT not in m.keys(): return False
    if INHIBITOR_KILLS not in m.keys(): return False
    if KILLING_SPREES not in m.keys(): return False
    if KILLS not in m.keys(): return False
    if LARGEST_CRITICAL_STRIKE not in m.keys(): return False
    if LARGEST_KILLING_SPREE not in m.keys(): return False
    if LARGEST_MULTI_KILL not in m.keys(): return False
    if MAGIC_DAMAGE_DEALT not in m.keys(): return False
    if MAGIC_DAMAGE_DEALT_TO_CHAMPIONS not in m.keys(): return False
    if MAGIC_DAMAGE_TAKEN not in m.keys(): return False
    if MINIONS_KILLED not in m.keys(): return False
    if NEUTRAL_MINIONS_KILLED not in m.keys(): return False
    if NEUTRAL_MINIONS_KILLED_ENEMY_JUNGLE not in m.keys(): return False
    if NEUTRAL_MINIONS_KILLED_TEAM_JUNGLE not in m.keys(): return False
    if PENTA_KILLS not in m.keys(): return False
    if PHYSICAL_DAMAGE_DEALT not in m.keys(): return False
    if PHYSICAL_DAMAGE_DEALT_TO_CHAMPIONS not in m.keys(): return False
    if PHYSICAL_DAMAGE_TAKEN not in m.keys(): return False
    if QUADRA_KILLS not in m.keys(): return False
    if SIGHT_WARDS_BOUGHT_IN_GAME not in m.keys(): return False
    if TOTAL_DAMAGE_DEALT not in m.keys(): return False
    if TOTAL_DAMAGE_DEALT_TO_CHAMPIONS not in m.keys(): return False
    if TOTAL_DAMAGE_TAKEN not in m.keys(): return False
    if TOTAL_HEAL not in m.keys(): return False
    if TOTAL_TIME_CROWD_CONTROL_DEALT not in m.keys(): return False
    if TOTAL_UNITS_HEALED not in m.keys(): return False
    if TOWER_KILLS not in m.keys(): return False
    if TRIPLE_KILLS not in m.keys(): return False
    if TRUE_DAMAGE_DEALT not in m.keys(): return False
    if TRUE_DAMAGE_DEALT_TO_CHAMPIONS not in m.keys(): return False
    if TRUE_DAMAGE_TAKEN not in m.keys(): return False
    if UNREAL_KILLS not in m.keys(): return False
    if VISION_WARDS_BOUGHT_IN_GAME not in m.keys(): return False
    if WARDS_KILLED not in m.keys(): return False
    if WARDS_PLACED not in m.keys(): return False
    return True

def init_player(p_dict, p_id, m):
    '''
    initialize the player's entry in the player_dict with the match data
    :param p_dict:
    :param m: match in a dict
    :return:
    '''
    m_id = m[MATCH_ID]
    p_dict[p_id] = {MATCH_ID_LIST : [m_id]}
    p_dict[p_id][MATCH_COUNT] = 1
    p_dict[p_id][FEATURES] = []
    p_dict[p_id][FEATURES].append(m[ASSISTS])
    p_dict[p_id][FEATURES].append(m[CHAMP_LEVEL])
    p_dict[p_id][FEATURES].append(m[COMBAT_PLAYER_SCORE])
    p_dict[p_id][FEATURES].append(m[DEATHS])
    p_dict[p_id][FEATURES].append(m[DOUBLE_KILLS])
    p_dict[p_id][FEATURES].append(m[FIRST_BLOOD_ASSIST])
    p_dict[p_id][FEATURES].append(m[FIRST_BLOOD_KILL])
    p_dict[p_id][FEATURES].append(m[FIRST_INHIBITOR_ASSIST])
    p_dict[p_id][FEATURES].append(m[FIRST_INHIBITOR_KILL])
    p_dict[p_id][FEATURES].append(m[FIRST_TOWER_ASSIST])
    p_dict[p_id][FEATURES].append(m[FIRST_TOWER_KILL])
    p_dict[p_id][FEATURES].append(m[GOLD_EARNED])
    p_dict[p_id][FEATURES].append(m[GOLD_SPENT])
    p_dict[p_id][FEATURES].append(m[INHIBITOR_KILLS])
    p_dict[p_id][FEATURES].append(m[KILLING_SPREES])
    p_dict[p_id][FEATURES].append(m[KILLS])
    p_dict[p_id][FEATURES].append(m[LARGEST_CRITICAL_STRIKE])
    p_dict[p_id][FEATURES].append(m[LARGEST_KILLING_SPREE])
    p_dict[p_id][FEATURES].append(m[LARGEST_MULTI_KILL])
    p_dict[p_id][FEATURES].append(m[MAGIC_DAMAGE_DEALT])
    p_dict[p_id][FEATURES].append(m[MAGIC_DAMAGE_DEALT_TO_CHAMPIONS])
    p_dict[p_id][FEATURES].append(m[MAGIC_DAMAGE_TAKEN])
    p_dict[p_id][FEATURES].append(m[MINIONS_KILLED])
    p_dict[p_id][FEATURES].append(m[NEUTRAL_MINIONS_KILLED])
    p_dict[p_id][FEATURES].append(m[NEUTRAL_MINIONS_KILLED_ENEMY_JUNGLE])
    p_dict[p_id][FEATURES].append(m[NEUTRAL_MINIONS_KILLED_TEAM_JUNGLE])
    p_dict[p_id][FEATURES].append(m[PENTA_KILLS])
    p_dict[p_id][FEATURES].append(m[PHYSICAL_DAMAGE_DEALT])
    p_dict[p_id][FEATURES].append(m[PHYSICAL_DAMAGE_DEALT_TO_CHAMPIONS])
    p_dict[p_id][FEATURES].append(m[PHYSICAL_DAMAGE_TAKEN])
    p_dict[p_id][FEATURES].append(m[QUADRA_KILLS])
    p_dict[p_id][FEATURES].append(m[SIGHT_WARDS_BOUGHT_IN_GAME])
    p_dict[p_id][FEATURES].append(m[TOTAL_DAMAGE_DEALT])
    p_dict[p_id][FEATURES].append(m[TOTAL_DAMAGE_DEALT_TO_CHAMPIONS])
    p_dict[p_id][FEATURES].append(m[TOTAL_DAMAGE_TAKEN])
    p_dict[p_id][FEATURES].append(m[TOTAL_HEAL])
    p_dict[p_id][FEATURES].append(m[TOTAL_TIME_CROWD_CONTROL_DEALT])
    p_dict[p_id][FEATURES].append(m[TOTAL_UNITS_HEALED])
    p_dict[p_id][FEATURES].append(m[TOWER_KILLS])
    p_dict[p_id][FEATURES].append(m[TRIPLE_KILLS])
    p_dict[p_id][FEATURES].append(m[TRUE_DAMAGE_DEALT])
    p_dict[p_id][FEATURES].append(m[TRUE_DAMAGE_DEALT_TO_CHAMPIONS])
    p_dict[p_id][FEATURES].append(m[TRUE_DAMAGE_TAKEN])
    p_dict[p_id][FEATURES].append(m[UNREAL_KILLS])
    p_dict[p_id][FEATURES].append(m[VISION_WARDS_BOUGHT_IN_GAME])
    p_dict[p_id][FEATURES].append(m[WARDS_KILLED])
    p_dict[p_id][FEATURES].append(m[WARDS_PLACED])

def update_player(p_dict, p_id, m):
    '''
    update the player's entry in the player_dict with the match data
    :param p_dict:
    :param m: match in a dict
    :return:
    '''
    m_id = m[MATCH_ID]
    p_dict[p_id][MATCH_ID_LIST].append(m_id)
    p_dict[p_id][MATCH_COUNT] += 1
    p_dict[p_id][FEATURES][0] += m[ASSISTS]
    p_dict[p_id][FEATURES][1] += m[CHAMP_LEVEL]
    p_dict[p_id][FEATURES][2] += m[COMBAT_PLAYER_SCORE]
    p_dict[p_id][FEATURES][3] += m[DEATHS]
    p_dict[p_id][FEATURES][4] += m[DOUBLE_KILLS]
    p_dict[p_id][FEATURES][5] += m[FIRST_BLOOD_ASSIST]
    p_dict[p_id][FEATURES][6] += m[FIRST_BLOOD_KILL]
    p_dict[p_id][FEATURES][7] += m[FIRST_INHIBITOR_ASSIST]
    p_dict[p_id][FEATURES][8] += m[FIRST_INHIBITOR_KILL]
    p_dict[p_id][FEATURES][9] += m[FIRST_TOWER_ASSIST]
    p_dict[p_id][FEATURES][10] += m[FIRST_TOWER_KILL]
    p_dict[p_id][FEATURES][11] += m[GOLD_EARNED]
    p_dict[p_id][FEATURES][12] += m[GOLD_SPENT]
    p_dict[p_id][FEATURES][13] += m[INHIBITOR_KILLS]
    p_dict[p_id][FEATURES][14] += m[KILLING_SPREES]
    p_dict[p_id][FEATURES][15] += m[KILLS]
    p_dict[p_id][FEATURES][16] += m[LARGEST_CRITICAL_STRIKE]
    p_dict[p_id][FEATURES][17] += m[LARGEST_KILLING_SPREE]
    p_dict[p_id][FEATURES][18] += m[LARGEST_MULTI_KILL]
    p_dict[p_id][FEATURES][19] += m[MAGIC_DAMAGE_DEALT]
    p_dict[p_id][FEATURES][20] += m[MAGIC_DAMAGE_DEALT_TO_CHAMPIONS]
    p_dict[p_id][FEATURES][21] += m[MAGIC_DAMAGE_TAKEN]
    p_dict[p_id][FEATURES][22] += m[MINIONS_KILLED]
    p_dict[p_id][FEATURES][23] += m[NEUTRAL_MINIONS_KILLED]
    p_dict[p_id][FEATURES][24] += m[NEUTRAL_MINIONS_KILLED_ENEMY_JUNGLE]
    p_dict[p_id][FEATURES][25] += m[NEUTRAL_MINIONS_KILLED_TEAM_JUNGLE]
    p_dict[p_id][FEATURES][26] += m[PENTA_KILLS]
    p_dict[p_id][FEATURES][27] += m[PHYSICAL_DAMAGE_DEALT]
    p_dict[p_id][FEATURES][28] += m[PHYSICAL_DAMAGE_DEALT_TO_CHAMPIONS]
    p_dict[p_id][FEATURES][29] += m[PHYSICAL_DAMAGE_TAKEN]
    p_dict[p_id][FEATURES][30] += m[QUADRA_KILLS]
    p_dict[p_id][FEATURES][31] += m[SIGHT_WARDS_BOUGHT_IN_GAME]
    p_dict[p_id][FEATURES][32] += m[TOTAL_DAMAGE_DEALT]
    p_dict[p_id][FEATURES][33] += m[TOTAL_DAMAGE_DEALT_TO_CHAMPIONS]
    p_dict[p_id][FEATURES][34] += m[TOTAL_DAMAGE_TAKEN]
    p_dict[p_id][FEATURES][35] += m[TOTAL_HEAL]
    p_dict[p_id][FEATURES][36] += m[TOTAL_TIME_CROWD_CONTROL_DEALT]
    p_dict[p_id][FEATURES][37] += m[TOTAL_UNITS_HEALED]
    p_dict[p_id][FEATURES][38] += m[TOWER_KILLS]
    p_dict[p_id][FEATURES][39] += m[TRIPLE_KILLS]
    p_dict[p_id][FEATURES][40] += m[TRUE_DAMAGE_DEALT]
    p_dict[p_id][FEATURES][41] += m[TRUE_DAMAGE_DEALT_TO_CHAMPIONS]
    p_dict[p_id][FEATURES][42] += m[TRUE_DAMAGE_TAKEN]
    p_dict[p_id][FEATURES][43] += m[UNREAL_KILLS]
    p_dict[p_id][FEATURES][44] += m[VISION_WARDS_BOUGHT_IN_GAME]
    p_dict[p_id][FEATURES][45] += m[WARDS_KILLED]
    p_dict[p_id][FEATURES][46] += m[WARDS_PLACED]

if __name__ == '__main__':
    main()