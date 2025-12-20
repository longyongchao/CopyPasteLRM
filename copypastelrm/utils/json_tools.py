"""
JSONL文件读取工具函数

提供读取JSONL（JSON Lines）格式文件的实用函数。
JSONL格式是每行一个JSON对象的文件格式，常用于机器学习数据集。
"""

import json
import logging
from pathlib import Path
from typing import Iterator, List, Dict, Any, Union, Optional

logger = logging.getLogger(__name__)

def read_json(file_path: Union[str, Path], field: str = None) -> Any:
    """
    读取JSON文件，返回解析后的对象

    Args:
        file_path: JSON文件路径

    Returns:
        解析后的Python对象（通常为dict或list）

    Raises:
        FileNotFoundError: 文件不存在
        json.JSONDecodeError: JSON解析错误
        Exception: 其他读取错误

    Example:
        >>> data = read_json("data.json")
        >>> print(data)
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"路径不是文件: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if field:
                if field not in data:
                    logger.error(f"json文件中不存在 {field} 字段: {e}")
                    raise
                else:
                    return data[field]
            return data
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析失败: {e}")
        raise
    except Exception as e:
        logger.error(f"读取文件时发生错误: {e}")
        raise


def read_jsonl(file_path: Union[str, Path]) -> Iterator[Dict[str, Any]]:
    """
    逐行读取JSONL文件，返回JSON对象的迭代器
    
    Args:
        file_path: JSONL文件路径
        
    Yields:
        Dict[str, Any]: 每行的JSON对象
        
    Raises:
        FileNotFoundError: 文件不存在
        json.JSONDecodeError: JSON解析错误
        Exception: 其他读取错误
        
    Example:
        >>> for item in read_jsonl("data.jsonl"):
        ...     print(item)
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    if not file_path.is_file():
        raise ValueError(f"路径不是文件: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                    
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"第{line_num}行JSON解析失败: {e}")
                    logger.warning(f"问题行内容: {line[:100]}...")
                    continue
                    
    except Exception as e:
        logger.error(f"读取文件时发生错误: {e}")
        raise


def read_jsonl_to_list(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    读取JSONL文件并返回所有JSON对象的列表
    
    Args:
        file_path: JSONL文件路径
        
    Returns:
        List[Dict[str, Any]]: 所有JSON对象的列表
        
    Raises:
        FileNotFoundError: 文件不存在
        Exception: 其他读取错误
        
    Example:
        >>> data = read_jsonl_to_list("data.jsonl")
        >>> print(f"共读取{len(data)}条记录")
    """
    return list(read_jsonl(file_path))


def read_jsonl_with_filter(
    file_path: Union[str, Path], 
    filter_func: Optional[callable] = None
) -> Iterator[Dict[str, Any]]:
    """
    读取JSONL文件并应用过滤函数
    
    Args:
        file_path: JSONL文件路径
        filter_func: 过滤函数，接收JSON对象作为参数，返回True保留，False过滤
        
    Yields:
        Dict[str, Any]: 过滤后的JSON对象
        
    Example:
        >>> # 只保留包含特定字段的记录
        >>> def has_name(item):
        ...     return 'name' in item
        >>> 
        >>> for item in read_jsonl_with_filter("data.jsonl", has_name):
        ...     print(item['name'])
    """
    for item in read_jsonl(file_path):
        if filter_func is None or filter_func(item):
            yield item

def save_json(
    data: Any, 
    file_path: Union[str, Path], 
    indent: Optional[int] = 4, 
    ensure_ascii: bool = False
) -> None:
    """
    将Python对象（通常为dict或list）保存为JSON文件。

    Args:
        data: 要保存的Python对象（必须是JSON可序列化的）。
        file_path: JSON文件保存路径。
        indent: 可选。如果为非None，则输出JSON将进行缩进美化，
                例如 indent=4 表示使用4个空格缩进。如果为 None，则紧凑输出。
        ensure_ascii: 可选。如果为 True，非ASCII字符将被转义；
                      如果为 False（默认中文友好），则直接输出Unicode字符。

    Raises:
        TypeError: 数据不可JSON序列化。
        Exception: 写入文件时发生其他错误。

    Example:
        >>> data_to_save = {"name": "张三", "age": 30}
        >>> save_json(data_to_save, "user_info.json")
    """
    file_path = Path(file_path)

    # 确保目标文件夹存在
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(
                data, 
                f, 
                indent=indent, 
                ensure_ascii=ensure_ascii
            )
        logger.info(f"数据已成功保存到: {file_path}")

    except TypeError as e:
        logger.error(f"数据类型不可JSON序列化: {e}")
        raise
    except Exception as e:
        logger.error(f"写入文件时发生错误: {e}")
        raise


def save_jsonl(data: List[Dict], file_path: str):
    """
    将字典列表保存为 JSONL 格式。
    
    参数:
    - data: 包含字典的列表 List[dict]
    - file_path: 保存的文件路径
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in data:
            # ensure_ascii=False 可以让中文正常显示，而不是显示为 \uXXXX
            json_record = json.dumps(entry, ensure_ascii=False)
            f.write(json_record + '\n')


