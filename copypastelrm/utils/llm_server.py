from openai import OpenAI
import time
import random
from typing import Optional, List, Dict, Any

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

    def batch_call(
        self,
        model_name: str,
        user_prompts: List[str],
        system_prompts: Optional[List[str]] = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        enable_thinking: bool = False,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        jitter: bool = True,
    ) -> List[str]:
        """
        批量调用模型进行推理，使用OpenAI兼容的批量推理接口
        如果服务器不支持批量推理，则回退到单个调用

        Args:
            model_name: 模型名称
            user_prompts: 用户提示列表
            system_prompts: 系统提示列表（可选，如果为None则全部使用空字符串）
            temperature: 温度参数
            top_p: top_p参数
            enable_thinking: 是否启用思考模式
            max_retries: 最大重试次数
            initial_delay: 初始延迟
            max_delay: 最大延迟
            backoff_factor: 退避因子
            jitter: 是否添加随机抖动

        Returns:
            List[str]: 推理结果列表，失败的位置返回空字符串
        """
        batch_size = len(user_prompts)
        if system_prompts is None:
            system_prompts = [""] * batch_size

        # 尝试使用批量推理API
        results = self._try_batch_inference(
            model_name=model_name,
            user_prompts=user_prompts,
            system_prompts=system_prompts,
            temperature=temperature,
            top_p=top_p,
            enable_thinking=enable_thinking,
            max_retries=max_retries,
            initial_delay=initial_delay,
            max_delay=max_delay,
            backoff_factor=backoff_factor,
            jitter=jitter,
        )

        # 如果批量推理失败，回退到单个调用
        if results is None:
            results = []
            for user_prompt, system_prompt in zip(user_prompts, system_prompts):
                result = self.call(
                    model_name=model_name,
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    top_p=top_p,
                    enable_thinking=enable_thinking,
                    max_retries=max_retries,
                    initial_delay=initial_delay,
                    max_delay=max_delay,
                    backoff_factor=backoff_factor,
                    jitter=jitter,
                )
                results.append(result)

        return results

    def _try_batch_inference(
        self,
        model_name: str,
        user_prompts: List[str],
        system_prompts: List[str],
        temperature: float,
        top_p: float,
        enable_thinking: bool,
        max_retries: int,
        initial_delay: float,
        max_delay: float,
        backoff_factor: float,
        jitter: bool,
    ) -> Optional[List[str]]:
        """
        尝试使用批量推理API（如果支持）
        返回None表示服务器不支持批量推理，需要回退到单个调用
        """
        last_exception = None
        current_delay = initial_delay

        # 构建批量请求的消息列表
        batch_messages = []
        for user_prompt, system_prompt in zip(user_prompts, system_prompts):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            batch_messages.append(messages)

        for attempt in range(max_retries + 1):
            try:
                # 尝试使用OpenAI兼容的批量推理接口
                # 注意：vLLM和部分OpenAI兼容服务器可能不支持批量推理
                # 这里使用并发方式模拟批量推理
                import concurrent.futures

                def make_request(messages):
                    try:
                        response = self.client.chat.completions.create(
                            model=model_name,
                            messages=messages,
                            temperature=temperature,
                            top_p=top_p,
                            extra_body={
                                "chat_template_kwargs": {"enable_thinking": enable_thinking},
                            },
                        )
                        return response.choices[0].message.content.strip()
                    except Exception as e:
                        return f"ERROR: {str(e)}"

                # 使用线程池并发处理批量请求
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(batch_messages), 8)) as executor:
                    futures = [executor.submit(make_request, messages) for messages in batch_messages]
                    # 按原始顺序获取结果
                    ordered_results = [future.result() for future in futures]

                return ordered_results

            except Exception as e:
                last_exception = e

                # 检查是否是致命错误（不支持批量推理）
                if not self._is_retryable_error(e):
                    # 服务器可能不支持批量推理，返回None让调用者回退到单个调用
                    return None

                if attempt == max_retries:
                    print(f"批量推理失败，已达到最大重试次数 {max_retries}: {e}")
                    return None

                delay = min(current_delay, max_delay)
                if jitter:
                    delay = delay * (0.5 + random.random() * 0.5)

                time.sleep(delay)
                current_delay *= backoff_factor

        return None