from swift.llm import DatasetMeta, ResponsePreprocessor, load_dataset, register_dataset


class CopyPasteQAPreprocessor(ResponsePreprocessor):
    def preprocess(self, row: dict):
        """Process HotpotQA row with proper type annotations.

        Args:
            row: Input row containing HotpotQA data with strongly typed context

        Returns:
            Processed row with query, response, and solution fields
        """
        row.update(
            {
                "solution": {
                    "context": row["context"],
                    "supporting_facts": row["supporting_facts"],
                    "answers": row["answer_candidates"],
                    "dataset": row["dataset"],
                    "id": row["id"],
                }
            }
        )
        return super().preprocess(row)  # type: ignore[return-value]


register_dataset(
    DatasetMeta(
        dataset_path="../../data/possiblePassAtKEqual0Subset/Qwen3-4B-Instruct-2507_2000_copypaste_hotpotqa-171_multirc-300_popqa-58_musiqua-105_2wiki-123.jsonl",
        dataset_name="Qwen3-4B-I_2000_copypaste",
        preprocess_func=CopyPasteQAPreprocessor(
            columns={
                "solution": "solution",
            }
        ),
    )
)

register_dataset(
    DatasetMeta(
        dataset_path="../../data/possiblePassAtKEqual0Subset/Qwen3-4B-Instruct-2507_2000_deepseek_hotpotqa-171_multirc-300_popqa-58_musiqua-105_2wiki-123.jsonl",
        dataset_name="Qwen3-4B-I_2000_deepseek",
        preprocess_func=CopyPasteQAPreprocessor(
            columns={
                "solution": "solution",
            }
        ),
    )
)

if __name__ == "__main__":
    dataset = load_dataset(
        ["Qwen3-4B-I_2000_copypaste"], remove_unused_columns=False, download_mode="force_redownload"
    )

    print(f"dataset: {dataset}")
    print(f"dataset[0]: {dataset[0]}")
    print(f"dataset[0][0]: {dataset[0][0]}")
