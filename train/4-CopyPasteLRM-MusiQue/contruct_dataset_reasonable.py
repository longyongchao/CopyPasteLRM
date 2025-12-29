from copypastelrm.utils.json_tools import read_jsonl_to_list, save_jsonl
import random


if __name__ == "__main__":

    data_path = f"key_data/reasonable_hard_subset/train/musique_answer_f1_128.jsonl"
    data = read_jsonl_to_list(data_path)

    print('Total MuSiQue IDs loaded:', len(data))
    # 过滤掉2跳的数据，只使用3跳及以上的数据
    data = [item for item in data if item['llm_response'] == 'yes' ]
    print('MuSiQue IDs after filtering unreasonable:', len(data))

    trainset = []
    
    for item in data:

        # --- 构建新的样本结构 ---
        trainset.append(
            {
                "id": item['id'],
            }
        )

    random.seed(42)
    random.shuffle(trainset)
    
    print(f"Final trainset size: {len(trainset)} samples")

    # 6. 保存最终合并的数据集
    # 输出路径包含模型名、重采样数、Prompt类型以及各数据集的具体数量信息
    output_path = 'train/4-CopyPasteLRM-MusiQue/trainset/Qwen3_4B_I-musique_128_without_2hop_reasonable.jsonl'
    save_jsonl(trainset, output_path)
    
    print(f"Successfully saved {len(trainset)} samples to: {output_path}")