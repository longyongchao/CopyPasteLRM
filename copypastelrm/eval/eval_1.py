from typing import List
from copypastelrm.utils.json_tools import read_json
import argparse
from tqdm import tqdm

import numpy as np
from copypastelrm.metrics.utils import extract_answer_and_facts, extract_answer_and_facts_old
from copypastelrm.metrics.HotpotQA import compute_answer_em_hit_f1, update_sp

def main(
    path: str,
):
    results = read_json(path)
    info = results['info']
    data = results['data']

    prompt_type = info['prompt_type']
    model_name = info['model_name']

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

    distribution = {}

    for sample_id, item in tqdm(data.items()):
        dataset = item['dataset']
        predict = item['predict']
        gold_answers: List[str] = item['answer']
        gold_facts = item['sfs']

        if predict is None:
            print(f'predict is None, dataset: {dataset}, sample_id: {sample_id}')
            continue

        if dataset not in distribution:
            distribution[dataset] = {
                'answer_em': [],
                'answer_hit': [],
                'answer_f1': [],
                'fact_f1': [],
                'fact_em': [],
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
        distribution[dataset]['answer_predict_length'].append(answer_predict_length)
        distribution[dataset]['answer_gold_length'].append(answer_gold_length)
        distribution[dataset]['fact_predict_length'].append(fact_predict_length)
        distribution[dataset]['fact_gold_length'].append(fact_gold_length)
        distribution[dataset]['predict_length'].append(predict_length)
        distribution[dataset]['answer_parser_success_rate'].append(answer_parser_success_rate)
        distribution[dataset]['fact_parser_success_rate'].append(fact_parser_success_rate)

    # print(distribution)
    metrics = {}
    for dataset, values in distribution.items():
        # 断言所有的value的长度均一致
        assert len(set([ len(value) for value in values.values() ])) == 1
        metrics[dataset] = {
            'total': len(values['answer_em']),
            'ans_em': np.mean(values['answer_em']),
            'ans_hit': np.mean(values['answer_hit']),
            'ans_f1': np.mean(values['answer_f1']),
            'fact_f1': np.mean(values['fact_f1']),
            'fact_em': np.mean(values['fact_em']),
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
        metric_name.extend(metric.keys())
        metric_value.extend(metric.values())
    
    print(f'[prompt_type: {prompt_type}]\n[model_name: {model_name}]\n\n')

    print(",".join(metric_name))
    print('\n')
    metric_value = [str(value) for value in metric_value]
    print(",".join(metric_value))

    print('\n\n\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)

    args = parser.parse_args()

    print('路径：', args.path)

    main(args.path)