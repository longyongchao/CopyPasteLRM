from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import List, Dict, Union

class ChatTokenCounter:
    def __init__(self, model_name_or_path: Union[str, PreTrainedTokenizer, PreTrainedTokenizerFast]):
        """
        初始化计数器。
        
        Args:
            model_name_or_path: 可以是模型路径(str)，也可以是已经加载好的 tokenizer 实例。
        """
        if isinstance(model_name_or_path, str):
            # 只有在传入字符串时才加载，避免外部传入实例时重复加载
            print(f"⏳ Loading tokenizer from {model_name_or_path} ...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, 
                trust_remote_code=True
            )
        else:
            self.tokenizer = model_name_or_path

        # 预检查模板，避免在循环中重复检查
        if not self.tokenizer.chat_template:
            if hasattr(self.tokenizer, "default_chat_template"):
                self.tokenizer.chat_template = self.tokenizer.default_chat_template
            else:
                print("⚠️ Warning: Tokenizer has no chat_template configured.")

    def __call__(self, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> int:
        """
        使类实例可以直接像函数一样被调用: counter(messages)
        """
        try:
            token_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt
            )
            return len(token_ids)
        except Exception as e:
            # 生产环境中，最好记录日志而不是直接 print
            print(f"Tokenization Error: {e}")
            return 0

    def get_token_ids(self, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> List[int]:
        """如果需要获取具体的 IDs"""
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt
        )
        
if __name__ == "__main__":

    counter = ChatTokenCounter('Qwen/Qwen2.5-3B-Instruct')

    input_messages = [
        {"role": "system", "content": "你是一个数学专家。"},
        {"role": "user", "content": "证明素数有无穷多个。"}
    ]

    num = counter(input_messages)

    print('token数量：', num)