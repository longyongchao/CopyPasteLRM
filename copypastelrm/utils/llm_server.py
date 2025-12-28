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
        prompt: str,
        temperature: float = 0.7,
        top_p=0.95,
        enable_thinking: bool = False,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        jitter: bool = True,
    ) -> str:
        """
        调用模型进行推理，支持指数退避重试

        Args:
            model_name: 模型名称
            prompt: 输入提示
            temperature: 温度参数
            top_p: top_p 参数
            enable_thinking: 是否启用思考模式
            max_retries: 最大重试次数
            initial_delay: 初始延迟时间（秒）
            max_delay: 最大延迟时间（秒）
            backoff_factor: 退避因子
            jitter: 是否添加随机抖动

        Returns:
            str: 模型生成的回答
        """
        last_exception = None
        current_delay = initial_delay
        
        for attempt in range(max_retries + 1):  # +1 包括第一次尝试
            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    top_p=top_p,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": enable_thinking},
                        "enable_thinking": enable_thinking,
                    },
                )
                print(response)
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                last_exception = e
                
                # 如果是最后一次尝试，直接抛出异常
                if attempt == max_retries:
                    print(f"模型调用失败，已达到最大重试次数 {max_retries}: {e}")
                    return ""
                
                # 判断是否为可重试的错误类型
                if self._is_retryable_error(e):
                    # 计算延迟时间
                    delay = min(current_delay, max_delay)
                    
                    # 添加随机抖动以避免雷群效应
                    if jitter:
                        delay = delay * (0.5 + random.random() * 0.5)
                    
                    print(f"模型调用失败 (尝试 {attempt + 1}/{max_retries + 1}): {e}")
                    print(f"等待 {delay:.2f} 秒后重试...")
                    
                    time.sleep(delay)
                    
                    # 指数退避
                    current_delay *= backoff_factor
                else:
                    # 不可重试的错误，直接返回
                    print(f"模型调用失败，不可重试的错误: {e}")
                    return ""
        
        # 理论上不会到达这里，但为了安全起见
        print(f"模型调用失败，未知错误: {last_exception}")
        return ""
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """
        判断错误是否可以重试
        
        Args:
            error: 异常对象
            
        Returns:
            bool: 是否可重试
        """
        error_str = str(error).lower()
        
        # 网络相关错误
        retryable_patterns = [
            "connection",
            "timeout",
            "network",
            "temporary",
            "rate limit",
            "too many requests",
            "service unavailable",
            "internal server error",
            "503",
            "502",
            "500",
            "429",
        ]
        
        return any(pattern in error_str for pattern in retryable_patterns)
