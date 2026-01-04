from openai import OpenAI
import time
import random
from typing import Optional

class LLMServer:
    def __init__(self, base_url: str = 'http://127.0.0.1:8124', api_key: str = "sk-xxxx"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def call(
        self,
        model_name: str,
        user_prompt: str,
        system_prompt: str = "",  # 新增：支持 system prompt，默认为空
        temperature: float = 0.7,
        top_p: float = 0.95,
        enable_thinking: bool = False,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        jitter: bool = True,
    ) -> str:
        """
        调用模型进行推理，支持指数退避重试和区分 system/user prompt
        """
        last_exception = None
        current_delay = initial_delay
        
        # 构建消息列表
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        for attempt in range(max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,  # 使用构建好的消息列表
                    temperature=temperature,
                    top_p=top_p,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": enable_thinking},
                        # 部分推理框架可能在顶层也需要此参数，视具体后端而定
                        # "enable_thinking": enable_thinking, 
                    },
                )
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                last_exception = e
                
                if attempt == max_retries:
                    print(f"模型调用失败，已达到最大重试次数 {max_retries}: {e}")
                    return ""
                
                if self._is_retryable_error(e):
                    delay = min(current_delay, max_delay)
                    if jitter:
                        delay = delay * (0.5 + random.random() * 0.5)
                    
                    # 简化日志输出，避免刷屏
                    # print(f"调用重试 ({attempt + 1}/{max_retries}): {e} | 等待 {delay:.2f}s")
                    
                    time.sleep(delay)
                    current_delay *= backoff_factor
                else:
                    print(f"模型调用失败，不可重试的错误: {e}")
                    return ""
        
        return ""
    
    def _is_retryable_error(self, error: Exception) -> bool:
        error_str = str(error).lower()
        retryable_patterns = [
            "connection", "timeout", "network", "temporary", "rate limit",
            "too many requests", "service unavailable", "internal server error",
            "503", "502", "500", "429", "remotedisconnected"
        ]
        return any(pattern in error_str for pattern in retryable_patterns)