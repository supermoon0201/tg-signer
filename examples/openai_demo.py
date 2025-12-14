"""OpenAI Client 测试脚本（支持文本对话和识图功能）

这是一个用于测试 OpenAI API 客户端功能的独立示例脚本。
支持自定义 base_url、model 和 api_key，方便测试代理或兼容 OpenAI API 的服务。

功能特性：
- 文本对话：支持流式和非流式输出
- 识图功能（选择题模式）：使用项目中的识图提示词，返回JSON格式结果
- 识图功能（通用模式）：描述和分析图片内容

使用示例：
    # 文本对话（基本使用）
    python examples/openai_demo.py --model gpt-4o-mini --prompt "你好"

    # 文本对话（流式输出）
    python examples/openai_demo.py --model gpt-4o-mini --stream --prompt "讲个笑话"

    # 识图功能（选择题模式）
    python examples/openai_demo.py --model gpt-4o --image test.jpg --prompt "这是什么动物？" --options "0:猫" "1:狗" "2:兔子"

    # 识图功能（通用模式）
    python examples/openai_demo.py --model gpt-4o --image test.jpg --prompt "请详细描述这张图片"

    # 指定所有参数
    python examples/openai_demo.py \
        --base-url http://localhost:8000/v1 \
        --model gpt-4o \
        --api-key sk-xxx \
        --image test.jpg \
        --prompt "识别图片内容" \
        --options "0:选项A" "1:选项B"

环境变量：
    OPENAI_BASE_URL: API 基础URL（可选）
    OPENAI_MODEL: 模型名称
    OPENAI_API_KEY: API密钥
    OPENAI_PROMPT: 默认提示词（可选）
    OPENAI_VISION_PROMPT: 识图模式的系统提示词（可选）
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
from pathlib import Path
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    print("[错误] 未安装 openai 库，请运行: pip install openai", file=sys.stderr)
    sys.exit(1)

try:
    import json_repair
except ImportError:
    print("[警告] 未安装 json_repair 库，将使用标准 json 解析", file=sys.stderr)
    json_repair = None

# 尝试从项目中导入 encode_image 函数
try:
    from tg_signer.ai_tools import encode_image
except ImportError:
    # 如果导入失败，使用本地实现
    import base64
    def encode_image(image_bytes: bytes) -> str:
        """将图片字节编码为base64字符串（备用实现）"""
        return base64.b64encode(image_bytes).decode("utf-8")

# 项目中的选择题识图提示词（来自 tg_signer/ai_tools.py:120-126）
CHOICE_VISION_PROMPT = """你是一个**图片识别助手**，可以根据提供的图片和问题选择出**唯一正确**的选项，如果你觉得每个都不对，也要给出一个你认为最符合的答案，以如下JSON格式输出你的回复：
    {
      "option": 1,  // 整数，表示选项的序号，从0开始。
      "reason": "这么选择的原因，30字以内"
    }
    option字段表示你选择的选项。
    """

# 通用识图提示词
GENERAL_VISION_PROMPT = """你是一个**图片识别助手**，可以根据提供的图片和问题给出详细的回答。
请仔细观察图片内容，结合用户的问题，给出准确、详细的描述和分析。"""

# 支持的图片格式（仅OpenAI API官方支持的格式）
SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".webp"]


def load_image_file(image_path: str) -> tuple[bytes, str]:
    """加载图片文件并返回字节数据和MIME类型

    Args:
        image_path: 图片文件路径

    Returns:
        (图片字节数据, MIME类型)

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 不支持的图片格式
    """
    path = Path(image_path)

    # 检查文件是否存在
    if not path.exists():
        raise FileNotFoundError(f"图片文件不存在: {image_path}")

    # 检查文件格式
    if path.suffix.lower() not in SUPPORTED_IMAGE_FORMATS:
        raise ValueError(
            f"不支持的图片格式: {path.suffix}\n"
            f"支持的格式: {', '.join(SUPPORTED_IMAGE_FORMATS)}"
        )

    # 读取文件
    with open(path, "rb") as f:
        image_bytes = f.read()

    # 推断MIME类型
    mime_type = mimetypes.guess_type(str(path))[0] or "image/jpeg"

    return image_bytes, mime_type


def parse_options(options_list: list[str]) -> list[tuple[int, str]]:
    """解析选项列表

    Args:
        options_list: 选项列表，格式如 ["0:猫", "1:狗"]

    Returns:
        解析后的选项列表 [(0, "猫"), (1, "狗")]

    Raises:
        ValueError: 选项格式错误
    """
    parsed_options = []
    for opt in options_list:
        if ":" not in opt:
            raise ValueError(f"选项格式错误: {opt}，应为 '序号:描述' 格式，如 '0:猫'")

        parts = opt.split(":", 1)
        try:
            index = int(parts[0])
            if index < 0:
                raise ValueError(f"选项序号必须为非负整数: {parts[0]}")
            description = parts[1].strip()
            if not description:
                raise ValueError(f"选项描述不能为空: {opt}")
            parsed_options.append((index, description))
        except ValueError as e:
            if "invalid literal" in str(e):
                raise ValueError(f"选项序号必须为整数: {parts[0]}")
            raise

    return parsed_options


def create_client(base_url: Optional[str], api_key: Optional[str]) -> OpenAI:
    """创建 OpenAI 客户端实例

    Args:
        base_url: 自定义 API 基础URL，如果为 None 则使用默认值
        api_key: API 密钥，如果为 None 则从环境变量读取

    Returns:
        配置好的 OpenAI 客户端实例
    """
    return OpenAI(
        base_url=base_url if base_url else None,
        api_key=api_key if api_key else None
    )


def execute_text_chat(
    client: OpenAI,
    model: str,
    prompt: str,
    stream: bool = False,
) -> None:
    """执行文本聊天请求并输出结果

    Args:
        client: OpenAI 客户端实例
        model: 模型名称
        prompt: 用户提示词
        stream: 是否启用流式输出

    Raises:
        RuntimeError: 请求失败或响应无效
    """
    print(f"[模式] 文本对话")
    print(f"[模型] {model}")
    print(f"[提示] {prompt}")
    print(f"[流式] {'是' if stream else '否'}")
    print("-" * 60)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=stream,
        )
    except Exception as exc:
        raise RuntimeError(f"请求失败: {exc}") from exc

    # 处理响应
    try:
        if stream:
            print("[响应] ", end="", flush=True)
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
            print("\n" + "-" * 60)
        else:
            if response.choices and response.choices[0].message.content:
                content = response.choices[0].message.content
                print(f"[响应] {content}")
                print("-" * 60)
            elif response.choices and response.choices[0].message.refusal:
                raise RuntimeError(f"模型拒绝响应: {response.choices[0].message.refusal}")
            else:
                raise RuntimeError("未收到有效响应内容")
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"处理响应失败: {exc}") from exc


def execute_vision_chat_choice(
    client: OpenAI,
    model: str,
    image_path: str,
    prompt: str,
    options: list[tuple[int, str]],
    stream: bool = False,
    use_json_mode: bool = True,
) -> None:
    """执行选择题识图请求并输出结果（与项目 choose_option_by_image 对齐）

    Args:
        client: OpenAI 客户端实例
        model: 模型名称
        image_path: 图片文件路径
        prompt: 用户问题
        options: 选项列表 [(0, "猫"), (1, "狗")]
        stream: 是否启用流式输出
        use_json_mode: 是否使用JSON模式（某些模型不支持）

    Raises:
        RuntimeError: 请求失败或响应无效
    """
    print(f"[模式] 识图对话（选择题模式）")
    print(f"[模型] {model}")
    print(f"[图片] {image_path}")
    print(f"[问题] {prompt}")
    print(f"[选项] {options}")
    print(f"[流式] {'是' if stream else '否'}")
    print("-" * 60)

    # 加载图片
    try:
        image_bytes, mime_type = load_image_file(image_path)
        base64_image = encode_image(image_bytes)
        data_url = f"data:{mime_type};base64,{base64_image}"
        print(f"[信息] 图片已加载，大小: {len(image_bytes)} 字节，类型: {mime_type}")
    except Exception as exc:
        raise RuntimeError(f"加载图片失败: {exc}") from exc

    # 构造消息（与项目保持一致）
    text_query = f"问题为：{prompt}, 选项为：{json.dumps(options, ensure_ascii=False)}。"
    messages = [
        {"role": "system", "content": CHOICE_VISION_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text_query},
                {
                    "type": "image_url",
                    "image_url": {"url": data_url}
                },
            ],
        },
    ]

    # 发送请求（与项目保持一致：response_format, temperature）
    try:
        # 构造请求参数
        request_params = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "temperature": 0.1,
        }

        # 仅在支持时添加 response_format
        if use_json_mode:
            request_params["response_format"] = {"type": "json_object"}
            print(f"[信息] 使用JSON模式")
        else:
            print(f"[信息] 不使用JSON模式（依赖提示词约束）")

        response = client.chat.completions.create(**request_params)
    except Exception as exc:
        raise RuntimeError(f"请求失败: {exc}") from exc

    # 处理响应
    try:
        if stream:
            # 流式模式：积累内容后解析
            print("[响应] 正在接收流式输出...\n")
            accumulated_content = ""
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    accumulated_content += content
                    print(content, end="", flush=True)
            print("\n" + "-" * 60)

            # 解析JSON
            result = parse_json_response(accumulated_content)
        else:
            # 非流式模式：直接解析
            if response.choices and response.choices[0].message.content:
                content = response.choices[0].message.content
                print(f"[原始响应] {content}")
                print("-" * 60)
                result = parse_json_response(content)
            elif response.choices and response.choices[0].message.refusal:
                raise RuntimeError(f"模型拒绝响应: {response.choices[0].message.refusal}")
            else:
                raise RuntimeError("未收到有效响应内容")

        # 显示解析结果
        display_choice_result(result, options)

    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"处理响应失败: {exc}") from exc


def execute_vision_chat_general(
    client: OpenAI,
    model: str,
    image_path: str,
    prompt: str,
    vision_prompt: str,
    stream: bool = False,
) -> None:
    """执行通用识图请求并输出结果

    Args:
        client: OpenAI 客户端实例
        model: 模型名称
        image_path: 图片文件路径
        prompt: 用户问题
        vision_prompt: 系统提示词
        stream: 是否启用流式输出

    Raises:
        RuntimeError: 请求失败或响应无效
    """
    print(f"[模式] 识图对话（通用模式）")
    print(f"[模型] {model}")
    print(f"[图片] {image_path}")
    print(f"[问题] {prompt}")
    print(f"[流式] {'是' if stream else '否'}")
    print("-" * 60)

    # 加载图片
    try:
        image_bytes, mime_type = load_image_file(image_path)
        base64_image = encode_image(image_bytes)
        data_url = f"data:{mime_type};base64,{base64_image}"
        print(f"[信息] 图片已加载，大小: {len(image_bytes)} 字节，类型: {mime_type}")
    except Exception as exc:
        raise RuntimeError(f"加载图片失败: {exc}") from exc

    # 构造消息
    messages = [
        {"role": "system", "content": vision_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": data_url}
                },
            ],
        },
    ]

    # 发送请求
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=stream,
        )
    except Exception as exc:
        raise RuntimeError(f"请求失败: {exc}") from exc

    # 处理响应
    try:
        if stream:
            print("[响应] ", end="", flush=True)
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
            print("\n" + "-" * 60)
        else:
            if response.choices and response.choices[0].message.content:
                content = response.choices[0].message.content
                print(f"[响应] {content}")
                print("-" * 60)
            elif response.choices and response.choices[0].message.refusal:
                raise RuntimeError(f"模型拒绝响应: {response.choices[0].message.refusal}")
            else:
                raise RuntimeError("未收到有效响应内容")
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"处理响应失败: {exc}") from exc


def parse_json_response(content: str) -> dict:
    """解析JSON响应

    Args:
        content: JSON字符串

    Returns:
        解析后的字典

    Raises:
        RuntimeError: JSON解析失败或格式不正确
    """
    try:
        # 优先使用 json_repair（更容错）
        if json_repair:
            result = json_repair.loads(content)
        else:
            result = json.loads(content)
    except Exception as exc:
        raise RuntimeError(f"JSON解析失败: {exc}\n原始内容: {content}") from exc

    # 验证字段
    if not isinstance(result, dict):
        raise RuntimeError(f"响应不是有效的JSON对象: {result}")

    if "option" not in result:
        raise RuntimeError(f"响应缺少 'option' 字段: {result}")

    if "reason" not in result:
        raise RuntimeError(f"响应缺少 'reason' 字段: {result}")

    # 验证类型
    try:
        option = int(result["option"])
        result["option"] = option
    except (ValueError, TypeError) as exc:
        raise RuntimeError(f"'option' 字段必须为整数: {result['option']}") from exc

    if not isinstance(result["reason"], str):
        raise RuntimeError(f"'reason' 字段必须为字符串: {result['reason']}")

    return result


def display_choice_result(result: dict, options: list[tuple[int, str]]) -> None:
    """显示选择题结果

    Args:
        result: 解析后的结果 {"option": 1, "reason": "..."}
        options: 选项列表 [(0, "猫"), (1, "狗")]
    """
    option_index = result["option"]
    reason = result["reason"]

    print("\n" + "=" * 60)
    print("识图结果（JSON格式）")
    print("=" * 60)
    print(f"[选择的选项序号] {option_index}")

    # 查找对应的选项描述
    option_text = None
    for idx, desc in options:
        if idx == option_index:
            option_text = desc
            break

    if option_text:
        print(f"[选项内容] {option_text}")
    else:
        print(f"[警告] 选项序号 {option_index} 不在提供的选项列表中")

    print(f"[选择理由] {reason}")
    print("=" * 60)


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    """解析命令行参数

    Args:
        argv: 命令行参数列表

    Returns:
        解析后的参数命名空间
    """
    parser = argparse.ArgumentParser(
        description="OpenAI API 客户端测试工具 - 支持文本对话和识图功能",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 文本对话
  %(prog)s --model gpt-4o-mini --prompt "你好"
  %(prog)s --model gpt-4o-mini --stream --prompt "讲个笑话"

  # 识图功能（选择题模式）
  %(prog)s --model gpt-4o --image test.jpg --prompt "这是什么动物？" --options "0:猫" "1:狗" "2:兔子"

  # 识图功能（通用模式）
  %(prog)s --model gpt-4o --image test.jpg --prompt "请详细描述这张图片"

注意：
  - 图片格式仅支持: jpg, jpeg, png, webp
  - 选择题模式需要提供 --options 参数
  - 选项格式: "序号:描述"，如 "0:猫" "1:狗"
        """
    )

    # API配置参数
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL"),
        help="自定义 API 基础URL（默认: 环境变量 OPENAI_BASE_URL 或 OpenAI 官方地址）"
    )

    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL"),
        help="模型名称（必需，默认: 环境变量 OPENAI_MODEL）"
    )

    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY"),
        help="API 密钥（必需，默认: 环境变量 OPENAI_API_KEY）"
    )

    # 对话参数
    parser.add_argument(
        "--prompt",
        default=os.getenv("OPENAI_PROMPT", "你好，请用一句话回复。"),
        help="用户提示词或问题（默认: 环境变量 OPENAI_PROMPT 或默认问候语）"
    )

    parser.add_argument(
        "--stream",
        action="store_true",
        help="启用流式输出"
    )

    # 识图功能参数
    parser.add_argument(
        "--image",
        type=str,
        help=f"图片文件路径（启用识图模式）。支持格式: {', '.join(SUPPORTED_IMAGE_FORMATS)}"
    )

    parser.add_argument(
        "--options",
        nargs="+",
        help='选择题选项列表（启用选择题模式）。格式: "序号:描述"，如 "0:猫" "1:狗"'
    )

    parser.add_argument(
        "--vision-prompt",
        type=str,
        default=os.getenv("OPENAI_VISION_PROMPT", GENERAL_VISION_PROMPT),
        help="识图模式的系统提示词（仅通用模式使用，默认: 环境变量 OPENAI_VISION_PROMPT）"
    )

    parser.add_argument(
        "--no-json-mode",
        action="store_true",
        help="禁用JSON模式（用于不支持 response_format 的模型，如某些第三方API）"
    )

    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    """主函数

    Args:
        argv: 命令行参数列表

    Returns:
        退出码（0表示成功，非0表示失败）
    """
    args = parse_arguments(argv)

    # 验证必需参数
    if not args.api_key:
        print(
            "[错误] 缺少 API 密钥。请通过 --api-key 参数或 OPENAI_API_KEY 环境变量提供。",
            file=sys.stderr
        )
        return 2

    if not args.model:
        print(
            "[错误] 缺少模型名称。请通过 --model 参数或 OPENAI_MODEL 环境变量提供。",
            file=sys.stderr
        )
        return 2

    # 显示配置信息
    print("=" * 60)
    print("OpenAI Client Demo - 配置信息")
    print("=" * 60)
    if args.base_url:
        # 截断过长的URL
        display_url = args.base_url if len(args.base_url) <= 50 else args.base_url[:47] + "..."
        print(f"[配置] Base URL: {display_url}")
    else:
        print(f"[配置] Base URL: 使用官方默认地址")
    print(f"[配置] Model: {args.model}")
    print(f"[配置] API Key: {args.api_key[:8]}...{args.api_key[-4:] if len(args.api_key) > 12 else '***'}")
    print("=" * 60)

    # 创建客户端
    try:
        client = create_client(args.base_url, args.api_key)
    except Exception as exc:
        print(f"\n[错误] 创建客户端失败: {exc}", file=sys.stderr)
        return 1

    # 根据参数选择执行模式
    try:
        if args.image:
            # 识图模式
            if args.options:
                # 选择题模式
                try:
                    options = parse_options(args.options)
                except ValueError as exc:
                    print(f"\n[错误] {exc}", file=sys.stderr)
                    return 2

                execute_vision_chat_choice(
                    client=client,
                    model=args.model,
                    image_path=args.image,
                    prompt=args.prompt,
                    options=options,
                    stream=args.stream,
                    use_json_mode=not args.no_json_mode
                )
            else:
                # 通用模式
                execute_vision_chat_general(
                    client=client,
                    model=args.model,
                    image_path=args.image,
                    prompt=args.prompt,
                    vision_prompt=args.vision_prompt,
                    stream=args.stream
                )
        else:
            # 文本对话模式
            execute_text_chat(
                client=client,
                model=args.model,
                prompt=args.prompt,
                stream=args.stream
            )

        print("[成功] 测试完成")
        return 0

    except KeyboardInterrupt:
        print("\n[信息] 用户中断", file=sys.stderr)
        return 130
    except RuntimeError as exc:
        print(f"\n[错误] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"\n[错误] 未预期的错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
