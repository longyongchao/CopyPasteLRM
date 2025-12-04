import os
import subprocess


def get_git_commit_id(short: bool = False) -> str:
    """
    获取当前 Git 仓库的 commit ID（修复 stdout 参数兼容问题）
    :param short: 是否返回简短版（7位）commit ID，默认返回完整40位
    :return: commit ID 字符串，失败返回空字符串
    """
    try:
        # 切换到当前脚本所在目录（确保在 Git 仓库内）
        repo_path = os.path.dirname(os.path.abspath(__file__))
        os.chdir(repo_path)

        # 构建 Git 命令：short 为 True 时加 --short
        cmd = ["git", "rev-parse"]
        if short:
            cmd.append("--short")
        cmd.append("HEAD")

        # 修复：移除 stdout 参数（check_output 自动捕获 stdout）
        result = subprocess.check_output(cmd, stderr=subprocess.PIPE, text=True, encoding="utf-8")  # 仅保留 stderr 重定向

        # 去除换行符，返回纯净的 commit ID
        return result.strip()

    except subprocess.CalledProcessError as e:
        # 打印具体错误（便于排查：如非 Git 仓库、无提交记录）
        print(f"Git 命令执行失败：{e.stderr.decode('utf-8').strip()}")
        return ""
    except FileNotFoundError:
        print("错误：系统未安装 Git，请先安装 Git")
        return ""
    except Exception as e:
        print(f"获取 commit ID 失败：{str(e)}")
        return ""


# 调用示例
if __name__ == "__main__":
    full_commit = get_git_commit_id()
    short_commit = get_git_commit_id(short=True)
    print(f"完整 commit ID：{full_commit}")
    print(f"简短 commit ID：{short_commit}")
