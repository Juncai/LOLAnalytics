import json
import Utils as util

tags_path = 'data/champ_tags.json'
output_path = 'data/champ_tags.pickle'

with open(tags_path, 'r') as f:
    # json_str = f.read().replace('\n', '')
    parsed_obj = json.load(f)


champ_dict = parsed_obj['data']

tag_champ_dict = {}
tag_set = set()


for c_name, champ in champ_dict.items():
    c_id = champ['id']
    tag_list = champ['tags']
    tag_champ_dict[c_id] = tag_list
    for t in tag_list:
        tag_set.add(t)

util.save((tag_set, tag_champ_dict), output_path)