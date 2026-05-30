"""验证 zzmeb 面板纯 HTTP 签到流程（无浏览器）：
RequestWebView 取 initData → POST /api/auth/telegram 换 session(cookie)
→ GET /api/user/profile 看是否已签 → POST /api/user/checkin 签到。
作者: le.yang
证明可用轻量 httpx 实现该面板签到（对应可扩展进 action 6 的新契约分支）。
"""
import asyncio
from urllib.parse import parse_qs, urlparse

import httpx
from pyrogram.raw.functions.messages import RequestWebView
from pyrogram.raw.functions.users import GetFullUser

from tg_signer.core import get_client

BOT = "zzmeb_bot"


async def get_init_data(app):
    bot_peer = await app.resolve_peer(BOT)
    uf = await app.invoke(GetFullUser(id=bot_peer))
    base = uf.full_user.bot_info.menu_button.url
    auth = await app.invoke(
        RequestWebView(peer=bot_peer, bot=bot_peer, platform="ios", url=base)
    )
    frag = urlparse(auth.url).fragment
    init_data = parse_qs(frag).get("tgWebAppData", [""])[0]
    parsed = urlparse(base)
    api_base = f"{parsed.scheme}://{parsed.netloc}"
    return api_base, init_data


async def http_checkin(api_base, init_data):
    """纯 httpx 复刻 zzmeb 面板签到流程，返回每一步结果。"""
    steps = []
    # cookies=True 让 client 自动保存 /api/auth/telegram 返回的 session cookie
    browser_headers = {
        "Origin": api_base,
        "Referer": api_base + "/",
        "User-Agent": (
            "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(
        base_url=api_base, timeout=30.0, follow_redirects=True, headers=browser_headers
    ) as client:
        # 1) 用 initData 换 session
        r = await client.post("/api/auth/telegram", json={"initData": init_data})
        auth = r.json()
        steps.append(("POST /api/auth/telegram", r.status_code, auth.get("ok"), auth.get("message")))
        if not auth.get("ok"):
            return steps, False

        # 2) 查 profile 看是否已签
        r = await client.get("/api/user/profile")
        prof = r.json().get("data", {}).get("profile", {})
        checked = prof.get("checkedInToday")
        steps.append(("GET /api/user/profile", r.status_code, prof.get("score"),
                      f"checkedInToday={checked}"))

        # 3) 已签则跳过（幂等），否则签到
        if checked:
            steps.append(("(skip)", "-", "-", "今日已签到，跳过"))
            return steps, True

        r = await client.post("/api/user/checkin")
        ck = r.json()
        steps.append(("POST /api/user/checkin", r.status_code, ck.get("ok"), ck.get("message")))
        return steps, bool(ck.get("ok"))


async def main():
    app = get_client(name="my_account", workdir="data")
    async with app:
        api_base, init_data = await get_init_data(app)
    print(f"[api_base] {api_base}")
    steps, ok = await http_checkin(api_base, init_data)
    print("\n=== 纯 HTTP 签到流程 ===")
    for path, status, val, msg in steps:
        print(f"  {path:32s} status={status} val={val} :: {msg}")
    print(f"\nRESULT: {'SUCCESS' if ok else 'FAILED'}")


if __name__ == "__main__":
    asyncio.run(main())

