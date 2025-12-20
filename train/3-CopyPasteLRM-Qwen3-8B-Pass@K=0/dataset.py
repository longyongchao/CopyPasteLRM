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
        dataset_path="../../passAtK0_data.jsonl",
        dataset_name="copypaste_qa",
        preprocess_func=CopyPasteQAPreprocessor(
            columns={
                "solution": "solution",
            }
        ),
    )
)

if __name__ == "__main__":
    dataset = load_dataset(
        ["copypaste_qa"], remove_unused_columns=False, download_mode="force_redownload"
    )

    print(f"dataset: {dataset}")
    print(f"dataset[0]: {dataset[0]}")
    print(f"dataset[0][0]: {dataset[0][0]}")
