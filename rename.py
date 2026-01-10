import os
import re
import argparse

def rename_files(root_dir, execute=False):
    """
    遍历目录，将文件名中的时间戳后缀去除。
    目标格式: dataset-noise_X-TIMESTAMP.json -> dataset-noise_X.json
    """
    # 正则解释:
    # ^(.*)             : 捕获组1，文件名的前半部分 (贪婪匹配)
    # (-[\d]{10,})      : 捕获组2，短横线后跟10位以上的数字 (匹配时间戳)
    # (\.json)$         : 捕获组3，扩展名
    pattern = re.compile(r"^(.*)(-[\d]{10,})(\.json)$")

    count = 0
    
    # os.walk 递归遍历所有子目录
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            match = pattern.match(filename)
            if match:
                base_name = match.group(1)   # 例如: 2wikimultihopqa-noise_0
                # timestamp = match.group(2) # 例如: -1767461353 (这部分被丢弃)
                extension = match.group(3)   # 例如: .json
                
                new_filename = f"{base_name}{extension}"
                
                old_path = os.path.join(dirpath, filename)
                new_path = os.path.join(dirpath, new_filename)
                
                if not execute:
                    print(f"[预览] {filename}  -->  {new_filename}")
                else:
                    try:
                        # 防止覆盖已存在的同名文件
                        if os.path.exists(new_path):
                            print(f"[跳过] 目标文件已存在: {new_filename}")
                        else:
                            os.rename(old_path, new_path)
                            print(f"[完成] {filename}  -->  {new_filename}")
                            count += 1
                    except OSError as e:
                        print(f"[错误] 重命名失败 {filename}: {e}")

    if execute:
        print(f"\n总计重命名了 {count} 个文件。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量移除文件名中的时间戳后缀")
    parser.add_argument("directory", type=str, help="需要处理的根目录路径")
    parser.add_argument("--run", action="store_true", help="确认执行重命名 (默认只打印预览)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.directory):
        print(f"错误: 目录 '{args.directory}' 不存在。")
        exit(1)

    print(f"正在扫描目录: {args.directory}")
    if not args.run:
        print("--- 当前为预览模式 (Dry Run)，不修改任何文件 ---")
        print("--- 请加上 --run 参数来实际执行 ---")
        print("-" * 50)
    
    rename_files(args.directory, execute=args.run)