"""探测 zzmeb 面板真实接口：复用现有 session，打开 WebApp 抓取网络请求。
作者: le.yang
只读为主：先加载页面看 info 接口；可选点击「立即签到」以发现 checkin 接口。
"""
import asyncio
import json
from urllib.parse import urlparse

from pyrogram.raw.functions.messages import RequestWebView
from pyrogram.raw.functions.users import GetFullUser

from tg_signer.core import get_client

BOT = "zzmeb_bot"
CLICK_SIGN = True  # 是否点击「立即签到」以发现 checkin 接口


async def get_auth_url(app):
    bot_peer = await app.resolve_peer(BOT)
    user_full = await app.invoke(GetFullUser(id=bot_peer))
    menu = user_full.full_user.bot_info.menu_button
    base_url = menu.url
    print(f"[菜单按钮URL] {base_url}")
    auth = await app.invoke(
        RequestWebView(peer=bot_peer, bot=bot_peer, platform="ios", url=base_url)
    )
    print(f"[鉴权URL] {auth.url[:120]}...")
    return auth.url


async def probe_page(auth_url):
    from playwright.async_api import async_playwright

    api_calls = []  # 记录所有 XHR/fetch 请求

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page()

            def on_request(req):
                if req.resource_type in ("xhr", "fetch"):
                    api_calls.append((req.method, req.url))

            page.on("request", on_request)

            print("[加载面板页面...]")
            await page.goto(auth_url, wait_until="networkidle", timeout=45000)
            await page.wait_for_timeout(3000)

            print("\n--- 页面加载阶段捕获的 API 请求 ---")
            for method, url in api_calls:
                path = urlparse(url).path
                print(f"  {method} {urlparse(url).netloc}{path}")

            if CLICK_SIGN:
                # 尝试点击「立即签到」以发现 checkin 接口
                before = len(api_calls)
                for label in ("立即签到", "签到", "立即簽到", "簽到"):
                    try:
                        btn = page.get_by_text(label, exact=False).first
                        await btn.wait_for(state="visible", timeout=4000)
                        await btn.click(timeout=4000)
                        print(f"\n[已点击「{label}」]")
                        await page.wait_for_timeout(3000)
                        break
                    except Exception:
                        continue
                print("\n--- 点击签到后新增的 API 请求 ---")
                for method, url in api_calls[before:]:
                    path = urlparse(url).path
                    print(f"  {method} {urlparse(url).netloc}{path}")

            # 输出所有去重后的接口路径，方便归纳
            print("\n--- 去重路径汇总 ---")
            seen = set()
            for method, url in api_calls:
                pu = urlparse(url)
                key = (method, pu.netloc, pu.path)
                if key in seen:
                    continue
                seen.add(key)
                print(f"  {method} {pu.netloc}{pu.path}")
        finally:
            await browser.close()


async def main():
    app = get_client(name="my_account", workdir="data")
    async with app:
        auth_url = await get_auth_url(app)
    await probe_page(auth_url)


if __name__ == "__main__":
    asyncio.run(main())

