from copypastelrm.utils.json_tools import read_jsonl_to_list, save_jsonl
import random


if __name__ == "__main__":

    ids_path = f"key_data/pass_at_k_equal_0_subset/train/answer_f1/musique_128.jsonl"
    ids = read_jsonl_to_list(ids_path)

    print('Total MuSiQue IDs loaded:', len(ids))
    # 过滤掉2跳的数据，只使用3跳及以上的数据
    ids = [_id for _id in ids if '2hop' not in _id ]
    print('MuSiQue IDs after filtering 2-hop:', len(ids))

    trainset = []
    
    for _id in ids:
        _id = str(_id) 

        # --- 构建新的样本结构 ---
        trainset.append(
            {
                "id": _id,
            }
        )

    random.seed(42)
    random.shuffle(trainset)
    
    print(f"Final trainset size: {len(trainset)} samples")

    # 6. 保存最终合并的数据集
    # 输出路径包含模型名、重采样数、Prompt类型以及各数据集的具体数量信息
    output_path = 'train/4-CopyPasteLRM-MusiQue/trainset/Qwen3_4B_I-musique_128_without_2hop.jsonl'
    save_jsonl(trainset, output_path)
    
    print(f"Successfully saved {len(trainset)} samples to: {output_path}")