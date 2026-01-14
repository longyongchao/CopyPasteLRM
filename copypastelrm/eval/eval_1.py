from typing import List
from copypastelrm.utils.json_tools import read_json, read_jsonl_to_list
import argparse

import numpy as np
from copypastelrm.metrics.utils import extract_answer_and_facts, extract_answer_and_facts_old
from copypastelrm.metrics.HotpotQA import compute_answer_em_hit_f1, update_sp
from copypastelrm.datasets import load, AvailableDataset


def get_pass_at_k_equal_0_subset_ids(pass_at_k_equal_0_subset_ids_paths: List[str]) -> List[int]:
    pass_at_k_equal_0_subset_ids_paths = [
        "key_data/reasonable_hard_subset/test/hotpotqa.jsonl",
        "key_data/reasonable_hard_subset/test/musique.jsonl",
        "key_data/pass_at_k_equal_0_subset/test/answer_em/faitheval_42_answer_em.jsonl",
        "key_data/pass_at_k_equal_0_subset/test/answer_em/multirc_42_answer_em.jsonl",
        "key_data/pass_at_k_equal_0_subset/test/answer_em/popqa_42_answer_em.jsonl",
        "key_data/pass_at_k_equal_0_subset/test/answer_em/pubmedqa_42_answer_em.jsonl",
        "key_data/pass_at_k_equal_0_subset/test/answer_em/qasper_42_answer_em.jsonl",
    ]

    subset_ids = set()

    for path in pass_at_k_equal_0_subset_ids_paths:
        ids = read_jsonl_to_list(path)
        ids = [str(id) for id in ids]
        subset_ids.update(ids)
    
    return list(subset_ids)

def main(
    path: str,
):
    results = read_json(path)
    info = results['info']
    data = results['data']

    prompt_type = info['prompt_type']
    model_name = info['model_name']
    dataset_name = info['dataset']

    samples = load(name=AvailableDataset(dataset_name), split='test', distractor_docs=0).dataset

    extract_answer_and_facts_func = extract_answer_and_facts_old if prompt_type in ['copypaste', 'deepseek'] else extract_answer_and_facts

    """需要统计每一个数据集的：
    1. answer_em
    2. answer_hit
    3. answer_f1
    4. fact_f1
    5. fact_em
    6. answer_predict_length
    7. answer_gold_length
    8. fact_predict_length
    9. fact_gold_length
    10. predict_length
    11. answer_parser_success_rate
    12. fact_parser_success_rate

    匹配不成功的使用None
    """

    pass_at_k_equal_0_subset_ids = get_pass_at_k_equal_0_subset_ids([])

    distribution = {}

    for sample_id, item in data.items():
        dataset = dataset_name
        predict = item['predict']
        gold_answers: List[str] = samples[sample_id]['answers']
        gold_facts = samples[sample_id]['facts']

        if predict is None:
            # print(f'predict is None, dataset: {dataset}, sample_id: {sample_id}')
            continue

        if dataset not in distribution:
            distribution[dataset] = {
                'answer_em': [],
                'answer_hit': [],
                'answer_f1': [],
                'fact_f1': [],
                'fact_em': [],
                'passAtkEqual0_answer_em': [],
                'passAtkEqual0_answer_hit': [],
                'passAtkEqual0_answer_f1': [],
                'passAtkEqual0_fact_f1': [],
                'passAtkEqual0_fact_em': [],
                'answer_predict_length': [],
                'answer_gold_length': [],
                'fact_predict_length': [],
                'fact_gold_length': [],
                'predict_length': [],
                'answer_parser_success_rate': [], 
                'fact_parser_success_rate': [],
            }
        
        
        predict_answer, predict_facts = extract_answer_and_facts_func(predict)
        answer_em, answer_hit, answer_f1 = compute_answer_em_hit_f1(predict_answer, gold_answers)
        fact_em, fact_f1, _, _ = update_sp(predict_facts, gold_facts)

        answer_predict_length = len(predict_answer) if predict_answer is not None else 0
        answer_gold_length = np.mean([ len(answer) for answer in gold_answers ])
        fact_predict_length = np.sum([ len(fact) for fact in predict_facts ])
        fact_gold_length = np.sum([ len(fact) for fact in gold_facts ])
        predict_length = len(predict)

        answer_parser_success_rate = 1 if predict_answer is not None else 0
        fact_parser_success_rate = 1 if predict_facts is not None and len(predict_facts) > 0 else 0

        distribution[dataset]['answer_em'].append(answer_em)
        distribution[dataset]['answer_hit'].append(answer_hit)
        distribution[dataset]['answer_f1'].append(answer_f1)
        distribution[dataset]['fact_f1'].append(fact_f1)
        distribution[dataset]['fact_em'].append(fact_em)

        if str(sample_id) in pass_at_k_equal_0_subset_ids:
            distribution[dataset]['passAtkEqual0_answer_em'].append(answer_em)
            distribution[dataset]['passAtkEqual0_answer_hit'].append(answer_hit)
            distribution[dataset]['passAtkEqual0_answer_f1'].append(answer_f1)
            distribution[dataset]['passAtkEqual0_fact_f1'].append(fact_f1)
            distribution[dataset]['passAtkEqual0_fact_em'].append(fact_em)

        distribution[dataset]['answer_predict_length'].append(answer_predict_length)
        distribution[dataset]['answer_gold_length'].append(answer_gold_length)
        distribution[dataset]['fact_predict_length'].append(fact_predict_length)
        distribution[dataset]['fact_gold_length'].append(fact_gold_length)
        distribution[dataset]['predict_length'].append(predict_length)
        distribution[dataset]['answer_parser_success_rate'].append(answer_parser_success_rate)
        distribution[dataset]['fact_parser_success_rate'].append(fact_parser_success_rate)

    metrics = {}
    for dataset, values in distribution.items():

        metrics[dataset] = {
            'total': len(values['answer_em']),
            'passAtkEqual0_total': len(values['passAtkEqual0_answer_em']),
            'ans_em': np.mean(values['answer_em']),
            'passAtkEqual0_ans_em': np.mean(values['passAtkEqual0_answer_em'] if len(values['passAtkEqual0_answer_em']) > 0 else [0]),
            'ans_hit': np.mean(values['answer_hit']),
            'passAtkEqual0_ans_hit': np.mean(values['passAtkEqual0_answer_hit'] if len(values['passAtkEqual0_answer_hit']) > 0 else [0]),
            'ans_f1': np.mean(values['answer_f1']),
            'passAtkEqual0_ans_f1': np.mean(values['passAtkEqual0_answer_f1'] if len(values['passAtkEqual0_answer_f1']) > 0 else [0]),
            'fact_f1': np.mean(values['fact_f1']),
            'passAtkEqual0_fact_f1': np.mean(values['passAtkEqual0_fact_f1'] if len(values['passAtkEqual0_fact_f1']) > 0 else [0]),
            'fact_em': np.mean(values['fact_em']),
            'passAtkEqual0_fact_em': np.mean(values['passAtkEqual0_fact_em'] if len(values['passAtkEqual0_fact_em']) > 0 else [0]),
            'ans_pred_len': np.mean(values['answer_predict_length']),
            'ans_gold_len': np.mean(values['answer_gold_length']),
            'fact_pred_len': np.mean(values['fact_predict_length']),
            'fact_gold_len': np.mean(values['fact_gold_length']),
            'pred_len': np.mean(values['predict_length']),
            'ans_parser_success_rate': np.mean(values['answer_parser_success_rate']),
            'fact_parser_success_rate': np.mean(values['fact_parser_success_rate']),
        }
    
    # 对metrics的key进行排序，按照名称的升序
    metrics = {k: metrics[k] for k in sorted(metrics.keys())}

    metric_name = []
    metric_value = []
    for dataset, metric in metrics.items():
        # print(dataset)
        metric_name.extend(metric.keys())
        metric_value.extend(metric.values())
    
    # print(f'[prompt_type: {prompt_type}]\n[model_name: {model_name}]\n\n')

    # print(",".join(metric_name))
    # print('\n')
    metric_value = [str(value) for value in metric_value]
    print(",".join(metric_value))

    # print('\n\n\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)

    args = parser.parse_args()

    # print('路径：', args.path)

    main(args.path)