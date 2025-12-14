"""Bark 通知发送模块

Bark 是一款iOS推送通知应用，支持通过HTTP API发送通知。
本模块提供异步发送Bark通知的功能。
"""

import logging
from typing import Optional
from urllib.parse import quote

from httpx import AsyncClient

logger = logging.getLogger("tg-signer")


async def bark_send(
    bark_url: str,
    title: str,
    body: str = "",
    *,
    sound: Optional[str] = None,
    group: Optional[str] = None,
) -> bool:
    """发送 Bark 通知

    Args:
        bark_url: Bark服务器完整URL，格式为 https://api.day.app/your_key
        title: 通知标题
        body: 通知内容
        sound: 通知声音，可选值如 'alarm', 'bell' 等
        group: 通知分组

    Returns:
        bool: 发送成功返回True，失败返回False

    Note:
        发送失败不会抛出异常，只会记录警告日志，避免影响主流程
    """
    if not bark_url or not bark_url.strip():
        logger.warning("Bark URL 为空，跳过发送通知")
        return False

    bark_url = bark_url.rstrip("/")

    # URL安全校验：防止SSRF攻击
    if not bark_url.startswith("https://"):
        logger.warning(f"Bark URL 必须使用 HTTPS 协议: {bark_url}")
        return False

    # 基本格式校验：URL应该包含设备key（至少有一个路径段）
    url_parts = bark_url.split("/")
    if len(url_parts) < 4:  # https://domain/key 至少4个部分
        logger.warning(f"Bark URL 格式不正确，缺少设备key: {bark_url}")
        return False

    # URL编码title和body
    encoded_title = quote(title, safe="")
    encoded_body = quote(body, safe="")

    # 构建完整的请求URL: {bark_url}/{title}/{body}
    request_url = f"{bark_url}/{encoded_title}/{encoded_body}"

    # 构建查询参数
    params = {}
    if sound:
        params["sound"] = sound
    if group:
        params["group"] = group

    try:
        async with AsyncClient(timeout=10.0) as client:
            response = await client.post(request_url, params=params)
            response.raise_for_status()

            result = response.json()
            if result.get("code") == 200:
                logger.debug(f"Bark 通知发送成功: {title}")
                return True
            else:
                logger.warning(
                    f"Bark 通知发送失败: code={result.get('code')}, "
                    f"message={result.get('message', '未知错误')}"
                )
                return False

    except Exception as e:
        logger.warning(f"Bark 通知发送异常: {e}", exc_info=False)
        return False
