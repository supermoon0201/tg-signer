import subprocess
import sys


def run_command(cmd):
    """Run a command and return its output."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=r"C:\Users\YangLe\Downloads\199.180.118.170\202601071101\tg-signer-yl",
        )
        print(f"Command: {cmd}")
        print(f"Return code: {result.returncode}")
        if result.stdout:
            print(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        print(f"Error running command: {e}")
        return -1, "", str(e)


if __name__ == "__main__":
    # Initialize git repository
    print("=== Initializing Git repository ===")
    run_command("git init")

    # Check status
    print("\n=== Checking Git status ===")
    run_command("git status")

    # Add all files
    print("\n=== Adding all files ===")
    run_command("git add .")

    # Check status again
    print("\n=== Checking Git status after add ===")
    run_command("git status")

    # Commit
    print("\n=== Creating commit ===")
    commit_msg = "fix(core): 添加 GIF 验证码识别异常处理，防止程序崩溃\n\n- ai_tools.py: 捕获 PIL.Image.open() 的 UnidentifiedImageError\n- core.py: 在 _choose_option_by_gif 中添加异常处理，识别失败时优雅跳过\n- 修复问题：GIF 验证码损坏导致程序崩溃，Docker 容器重启，签到任务重复执行"

    # For Windows, we need to handle the multi-line commit message carefully
    run_command(f'git commit -m "fix(core): 添加 GIF 验证码识别异常处理，防止程序崩溃"')

    # Check log
    print("\n=== Checking commit log ===")
    run_command("git log --oneline -3")
