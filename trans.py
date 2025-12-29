from copypastelrm.utils.json_tools import read_json, read_jsonl_to_list, save_jsonl

path = "key_data/hard/avaible_pass@K=0_subset_musique.jsonl"

data = read_jsonl_to_list(path)

ids = [str(item['id']) for item in data if item['llm_response'].lower().strip() == 'yes']

save_jsonl(ids, 'key_data/reasonable_hard_subset/musique.jsonl')