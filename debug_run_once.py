#!/usr/bin/env python
"""
临时调试脚本，用于在 PyCharm 中调试 tg-signer run-once 命令
使用方法：
1. 在 PyCharm 中创建 Run Configuration
2. Script path: 指向这个文件
3. 不需要设置 Parameters
4. 在需要的地方设置断点
5. 点击 Debug 运行
"""
import sys

# 确保项目根目录在 Python 路径中
import os
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tg_signer.cli import tg_signer

if __name__ == '__main__':
    # 模拟命令行参数：tg-signer run-once nebula
    sys.argv = ['tg-signer', 'run-once', 'nebula']

    # 如果需要调试其他命令，修改上面的 sys.argv
    # 例如：sys.argv = ['tg-signer', 'run-once', 'other_task', '-n', '100']

    sys.exit(tg_signer())
