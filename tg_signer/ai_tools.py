import base64
import json
import os
import pathlib
from typing import TYPE_CHECKING, Union

import json_repair
from pydantic import TypeAdapter
from typing_extensions import Optional, Required, TypedDict

if TYPE_CHECKING:
    from openai import AsyncOpenAI  # 在性能弱的机器上导入openai包实在有些慢

from tg_signer.utils import UserInput, print_to_user

DEFAULT_MODEL = "gpt-4o"


def encode_image(image: bytes):
    return base64.b64encode(image).decode("utf-8")


class OpenAIConfig(TypedDict, total=False):
    api_key: Required[str]
    base_url: Optional[str]
    model: Optional[str]


class OpenAIConfigManager:
    def __init__(self, workdir: Union[str, pathlib.Path]):
        self.workdir = pathlib.Path(workdir)

    def get_config_file(self) -> pathlib.Path:
        return self.workdir / ".openai_config.json"

    def has_env_config(self):
        return bool(os.environ.get("OPENAI_API_KEY"))

    def has_config(self) -> bool:
        return self.has_env_config() and bool(self.load_file_config())

    def load_file_config(self) -> Optional[dict]:
        config_file = self.get_config_file()
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as fp:
                c = json.load(fp)
            return TypeAdapter(OpenAIConfig).validate_python(c)
        return None

    def save_config(self, api_key: str, base_url: str = None, model: str = None):
        config_file = self.get_config_file()
        config = OpenAIConfig(api_key=api_key, base_url=base_url, model=model)
        with open(config_file, "w", encoding="utf-8") as fp:
            json.dump(config, fp, ensure_ascii=False, indent=2)

    def load_config(self) -> Optional[OpenAIConfig]:
        # 环境变量优先
        if self.has_env_config():
            return OpenAIConfig(
                api_key=os.environ["OPENAI_API_KEY"],
                base_url=os.environ.get("OPENAI_BASE_URL"),
                model=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL),
            )
        return self.load_file_config()

    def ask_for_config(self):
        print_to_user("开始配置OpenAI API并保存至本地。")
        input_ = UserInput()
        api_key = input_("请输入 OPENAI_API_KEY: ").strip()
        while not api_key:
            print_to_user("API Key不能为空！")
            api_key = input_("请输入 OPENAI_API_KEY: ").strip()

        base_url = (
            input_(
                "请输入 OPENAI_BASE_URL (可选，直接回车使用默认OpenAI地址): "
            ).strip()
            or None
        )
        model = (
            input_(
                f"请输入 OPENAI_MODEL (可选，直接回车使用默认模型({DEFAULT_MODEL})): "
            ).strip()
            or None
        )
        self.save_config(api_key, base_url=base_url, model=model)
        print_to_user("OpenAI配置已保存。")
        return self.load_config()


def get_openai_client(
    api_key: str = None,
    base_url: str = None,
    **kwargs,
) -> Optional["AsyncOpenAI"]:
    from openai import AsyncOpenAI, OpenAIError

    try:
        return AsyncOpenAI(api_key=api_key, base_url=base_url, **kwargs)
    except OpenAIError:
        return None


class AITools:
    def __init__(self, cfg: OpenAIConfig):
        self.client = get_openai_client(
            api_key=cfg["api_key"], base_url=cfg.get("base_url")
        )
        self.default_model = cfg.get("model") or DEFAULT_MODEL

    async def choose_option_by_image(
        self,
        image: bytes,
        query: str,
        options: list[tuple[int, str]],
        client: "AsyncOpenAI" = None,
        model: str = None,
        temperature=0.1,
    ) -> int:
        sys_prompt = """你是一个**图片识别助手**，可以根据提供的图片和问题选择出**唯一正确**的选项，如果你觉得每个都不对，也要给出一个你认为最符合的答案，以如下JSON格式输出你的回复：
    {
      "option": 1,  // 整数，表示选项的序号，从0开始。
      "reason": "这么选择的原因，30字以内"
    }
    option字段表示你选择的选项。
    """
        client = client or self.client
        model = model or self.default_model
        text_query = f"问题为：{query}, 选项为：{json.dumps(options)}。"
        messages = [
            {"role": "system", "content": sys_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_query},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image(image)}"
                        },
                    },
                ],
            },
        ]
        # noinspection PyTypeChecker
        completion = await client.chat.completions.create(
            messages=messages,
            model=model,
            response_format={"type": "json_object"},
            stream=False,
            temperature=temperature,
        )
        message = completion.choices[0].message
        result = json_repair.loads(message.content)
        return int(result["option"])

    async def recognize_gif_code(
        self,
        gif_bytes: bytes,
        options: list[str],
        client: "AsyncOpenAI" = None,
        model: str = None,
        temperature=0.1,
    ) -> int:
        from io import BytesIO
        from PIL import Image, UnidentifiedImageError
        import logging

        logger = logging.getLogger(__name__)

        # 验证下载的字节数据
        if not gif_bytes:
            raise ValueError("下载的 GIF 数据为空")

        logger.info(f"GIF 文件大小: {len(gif_bytes)} 字节")

        # 检查文件头以识别格式
        file_header = gif_bytes[:16].hex() if len(gif_bytes) >= 16 else gif_bytes.hex()
        logger.info(f"文件头 (hex): {file_header}")

        # 保存文件用于调试
        try:
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as tmp:
                tmp.write(gif_bytes)
                tmp_path = tmp.name
            logger.info(f"已保存原始文件到: {tmp_path}")
        except Exception as e:
            logger.warning(f"无法保存调试文件: {e}")

        # 尝试使用PIL打开图片
        frames = []
        try:
            gif = Image.open(BytesIO(gif_bytes))
            logger.info(f"成功打开图片，格式: {gif.format}, 大小: {gif.size}")

            # 提取所有帧
            try:
                while True:
                    frames.append(gif.copy().convert("RGBA"))
                    gif.seek(gif.tell() + 1)
            except EOFError:
                pass
            except Exception as e:
                logger.warning(f"提取帧时出错: {e}")
                if not frames:
                    frames = []

        except UnidentifiedImageError as e:
            # PIL无法识别，尝试作为视频处理
            logger.warning(f"PIL无法识别图片格式，尝试作为视频处理: {e}")

            try:
                import cv2
                import numpy as np

                # 将字节数据写入临时文件
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                    tmp.write(gif_bytes)
                    video_path = tmp.name

                logger.info(f"尝试使用OpenCV读取视频: {video_path}")

                # 使用OpenCV读取视频
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    raise ValueError("无法打开视频文件")

                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                logger.info(f"视频总帧数: {frame_count}")

                # 提取关键帧
                num_frames = min(5, frame_count)
                if num_frames > 1:
                    frame_indices = [
                        i * (frame_count - 1) // (num_frames - 1) for i in range(num_frames)
                    ]
                else:
                    frame_indices = [0]

                for idx in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        # 转换BGR到RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # 转换为PIL Image
                        pil_image = Image.fromarray(frame_rgb).convert("RGBA")
                        frames.append(pil_image)

                cap.release()

                # 清理临时文件
                try:
                    os.unlink(video_path)
                except:
                    pass

                logger.info(f"成功从视频提取 {len(frames)} 帧")

            except ImportError:
                raise ValueError(
                    f"无法识别 GIF 图片 (大小: {len(gif_bytes)} 字节, 文件头: {file_header})，"
                    f"且未安装 opencv-python 库来处理视频格式。请安装: pip install opencv-python"
                ) from e
            except Exception as video_error:
                raise ValueError(
                    f"无法识别 GIF 图片 (大小: {len(gif_bytes)} 字节, 文件头: {file_header})，"
                    f"尝试作为视频处理也失败: {video_error}"
                ) from e

        # 验证是否成功提取帧
        if not frames:
            raise ValueError("无法从图片/视频中提取任何帧")

        logger.info(f"成功提取 {len(frames)} 帧")

        # 合成帧为单张图片
        base_frame = frames[0]
        composite = Image.new("RGBA", base_frame.size, (255, 255, 255, 255))

        # 选择关键帧（最多5帧）
        num_frames = min(5, len(frames))
        if num_frames > 1:
            frame_indices = [
                i * (len(frames) - 1) // (num_frames - 1) for i in range(num_frames)
            ]
        else:
            frame_indices = [0]

        alpha_per_frame = 255 // num_frames

        for idx in frame_indices:
            frame = frames[idx].copy()
            frame.putalpha(alpha_per_frame)
            composite = Image.alpha_composite(composite, frame)

        # 转换为RGB用于识别
        final_image = composite.convert("RGB")
        buffer = BytesIO()
        final_image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

        logger.info(f"合成图片大小: {len(image_bytes)} 字节")

        sys_prompt = """你是一个**验证码识别助手**。用户会给你一张图片（可能是GIF的帧合成），图片中包含一个验证码文本。
你需要：
1. 识别出图片中的验证码文本
2. 从给定的选项中选择与验证码最匹配的选项

以如下JSON格式输出你的回复：
{
  "code": "识别出的验证码",
  "option": 0,  // 整数，表示选项的序号，从0开始
  "reason": "选择原因，30字以内"
}
"""
        client = client or self.client
        model = model or self.default_model
        text_query = f"请识别图片中的验证码，并从以下选项中选择最匹配的：{json.dumps(list(enumerate(options)), ensure_ascii=False)}"

        messages = [
            {"role": "system", "content": sys_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_query},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encode_image(image_bytes)}"
                        },
                    },
                ],
            },
        ]
        # noinspection PyTypeChecker
        completion = await client.chat.completions.create(
            messages=messages,
            model=model,
            response_format={"type": "json_object"},
            stream=False,
            temperature=temperature,
        )
        message = completion.choices[0].message
        result = json_repair.loads(message.content)
        return int(result["option"])

    async def calculate_problem(
        self,
        query: str,
        client: "AsyncOpenAI" = None,
        model: str = None,
        temperature=0.1,
    ) -> str:
        sys_prompt = """你是一个**答题助手**，可以根据用户的问题给出正确的回答，只需要回复答案，不要解释，不要输出任何其他内容。"""
        model = model or self.default_model
        client = client or self.client
        text = f"问题是: {query}\n\n只需要给出答案，不要解释，不要输出任何其他内容。The answer is:"
        # noinspection PyTypeChecker
        completion = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": text},
            ],
            model=model,
            stream=False,
            temperature=temperature,
        )
        return completion.choices[0].message.content.strip()

    async def get_reply(
        self,
        prompt: str,
        query: str,
        client: "AsyncOpenAI" = None,
        model: str = None,
    ) -> str:
        model = model or self.default_model
        client = client or self.client
        messages = [
            {
                "role": "system",
                "content": prompt,
            },
            {"role": "user", "content": f"{query}"},
        ]
        # noinspection PyTypeChecker
        completion = await client.chat.completions.create(
            messages=messages,
            model=model,
            stream=False,
        )
        message = completion.choices[0].message
        return message.content
