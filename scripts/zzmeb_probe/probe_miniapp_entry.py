"""探测 zzmeb_bot 的 MiniApp 入口类型，并尝试拿到可用的 WebView 认证链接。

作者: le.yang

目标：
1. 读取 /start 回复中的按钮结构，确认 MiniApp 入口到底是 callback / web_app / url 哪一种。
2. 对 `t.me/<bot>/<short_name>` 这类入口，分别尝试：
   - 直接走 RequestWebView(url=原始按钮URL)
   - 先构造 InputBotAppShortName，再走 GetBotApp / RequestAppWebView
3. 只读输出，不点击签到，不修改远端状态。
"""

import asyncio
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

from pyrogram.raw.functions.messages import GetBotApp, RequestAppWebView, RequestWebView
from pyrogram.raw.types import InputBotAppShortName
from pyrogram.types import InlineKeyboardMarkup

from tg_signer.core import get_client

BOT_USERNAME = "zzmeb_bot"
BOT_CHAT_ID = 8922873363
START_COMMAND = "/start"


@dataclass
class ProbeResult:
    """记录单次探测结果，便于最终统一打印。"""

    step: str
    ok: bool
    detail: str


def extract_short_name(raw_url: Optional[str]) -> Optional[str]:
    """从 t.me/<bot>/<short_name> 或 telegram.me/<bot>/<short_name> 中提取 short_name。"""
    if not raw_url:
        return None
    parsed = urlparse(raw_url)
    if parsed.netloc not in {"t.me", "telegram.me", "www.t.me", "www.telegram.me"}:
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        return None
    return parts[1]


async def fetch_latest_start_menu(app):
    """向 bot 发送 /start，并返回最新一条带内联键盘的回复消息。"""
    sent = await app.send_message(BOT_CHAT_ID, START_COMMAND)
    await asyncio.sleep(2)
    async for msg in app.get_chat_history(BOT_CHAT_ID, limit=5):
        if msg.id <= sent.id:
            break
        if isinstance(getattr(msg, "reply_markup", None), InlineKeyboardMarkup):
            return msg
    return None


def find_miniapp_button(message):
    """从 /start 回复里定位 MiniApp 按钮。"""
    reply_markup = getattr(message, "reply_markup", None)
    if not isinstance(reply_markup, InlineKeyboardMarkup):
        return None
    for row in reply_markup.inline_keyboard:
        for btn in row:
            text = getattr(btn, "text", "") or ""
            if "MiniApp" in text or "miniapp" in text:
                return btn
    return None


async def try_request_webview(app, chat_peer, bot_peer, raw_url: str) -> ProbeResult:
    """沿用当前仓库思路，直接对按钮 URL 调 RequestWebView。"""
    try:
        auth = await app.invoke(
            RequestWebView(
                peer=chat_peer,
                bot=bot_peer,
                platform="ios",
                url=raw_url,
            )
        )
        return ProbeResult(
            step="RequestWebView(url=raw_url)",
            ok=True,
            detail=auth.url[:300],
        )
    except Exception as exc:
        return ProbeResult(
            step="RequestWebView(url=raw_url)",
            ok=False,
            detail=str(exc),
        )


async def try_request_app_webview(app, chat_peer, bot_peer, short_name: str) -> list[ProbeResult]:
    """对 bot short_name 走 GetBotApp + RequestAppWebView。"""
    results: list[ProbeResult] = []
    input_app = InputBotAppShortName(bot_id=bot_peer, short_name=short_name)

    try:
        bot_app = await app.invoke(GetBotApp(app=input_app, hash=0))
        app_obj = getattr(bot_app, "app", None)
        title = getattr(app_obj, "title", None)
        results.append(
            ProbeResult(
                step="GetBotApp(short_name)",
                ok=True,
                detail=f"title={title!r}, short_name={short_name!r}",
            )
        )
    except Exception as exc:
        results.append(
            ProbeResult(
                step="GetBotApp(short_name)",
                ok=False,
                detail=str(exc),
            )
        )
        return results

    try:
        auth = await app.invoke(
            RequestAppWebView(
                peer=chat_peer,
                app=input_app,
                platform="ios",
            )
        )
        results.append(
            ProbeResult(
                step="RequestAppWebView(short_name)",
                ok=True,
                detail=auth.url[:300],
            )
        )
    except Exception as exc:
        results.append(
            ProbeResult(
                step="RequestAppWebView(short_name)",
                ok=False,
                detail=str(exc),
            )
        )
    return results


async def main():
    app = get_client(name="my_account", workdir="data")
    results: list[ProbeResult] = []

    async with app:
        message = await fetch_latest_start_menu(app)
        if not message:
            print("[FAIL] 未找到 /start 的带按钮回复消息")
            return 1

        button = find_miniapp_button(message)
        if not button:
            print("[FAIL] 未找到 MiniApp 按钮")
            return 1

        raw_url = getattr(button, "url", None)
        short_name = extract_short_name(raw_url)
        bot_peer = await app.resolve_peer(BOT_USERNAME)
        chat_peer = await app.resolve_peer(BOT_CHAT_ID)

        results.append(
            ProbeResult(
                step="Button Shape",
                ok=True,
                detail=(
                    f"text={button.text!r}, "
                    f"callback={bool(getattr(button, 'callback_data', None))}, "
                    f"web_app={bool(getattr(button, 'web_app', None))}, "
                    f"url={raw_url!r}, "
                    f"short_name={short_name!r}"
                ),
            )
        )

        if raw_url:
            results.append(await try_request_webview(app, chat_peer, bot_peer, raw_url))
        else:
            results.append(
                ProbeResult(
                    step="RequestWebView(url=raw_url)",
                    ok=False,
                    detail="按钮没有 url，无法探测",
                )
            )

        if short_name:
            results.extend(
                await try_request_app_webview(app, chat_peer, bot_peer, short_name)
            )
        else:
            results.append(
                ProbeResult(
                    step="RequestAppWebView(short_name)",
                    ok=False,
                    detail="未能从按钮 URL 提取 short_name",
                )
            )

    print("\n==== zzmeb miniapp 入口探测 ====")
    for result in results:
        flag = "PASS" if result.ok else "FAIL"
        print(f"[{flag}] {result.step}: {result.detail}")
    print("================================")

    return 0 if any(r.step == "RequestAppWebView(short_name)" and r.ok for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
