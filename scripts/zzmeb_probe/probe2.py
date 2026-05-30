"""探测 zzmeb 面板真实接口 v2：抓请求头(认证方式)+响应体(格式)+页面按钮+截图。
作者: le.yang
"""
import asyncio
import json
import os
from urllib.parse import urlparse

from pyrogram.raw.functions.messages import RequestWebView
from pyrogram.raw.functions.users import GetFullUser

from tg_signer.core import get_client

BOT = "zzmeb_bot"
OUT = os.environ.get("ZZMEB_PROBE_OUT", os.path.dirname(os.path.abspath(__file__)))


async def get_auth_url(app):
    bot_peer = await app.resolve_peer(BOT)
    user_full = await app.invoke(GetFullUser(id=bot_peer))
    base_url = user_full.full_user.bot_info.menu_button.url
    auth = await app.invoke(
        RequestWebView(peer=bot_peer, bot=bot_peer, platform="ios", url=base_url)
    )
    return auth.url


async def probe(auth_url):
    from playwright.async_api import async_playwright

    records = []  # (method, url, req_headers, status, resp_json/text)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            pending = {}

            def on_request(req):
                if req.resource_type in ("xhr", "fetch"):
                    pending[req] = dict(req.headers)

            async def on_response(resp):
                req = resp.request
                if req not in pending:
                    return
                body = None
                try:
                    body = await resp.json()
                except Exception:
                    try:
                        body = (await resp.text())[:300]
                    except Exception:
                        body = None
                records.append(
                    {
                        "method": req.method,
                        "url": resp.url,
                        "auth_headers": {
                            k: v
                            for k, v in pending[req].items()
                            if k.lower()
                            in ("authorization", "x-initdata", "cookie", "x-telegram-init-data", "x-init-data")
                        },
                        "status": resp.status,
                        "body": body,
                    }
                )

            page.on("request", on_request)
            page.on("response", lambda r: asyncio.create_task(on_response(r)))

            await page.goto(auth_url, wait_until="networkidle", timeout=45000)
            await page.wait_for_timeout(3500)

            # 列出页面上所有按钮文本
            btn_texts = await page.eval_on_selector_all(
                "button, [role=button], a",
                "els => els.map(e => (e.innerText||'').trim()).filter(Boolean).slice(0,40)",
            )
            await page.screenshot(path=f"{OUT}/zzmeb_panel.png", full_page=True)

            print("\n=== 捕获的 API 请求/响应 ===")
            for r in records:
                pu = urlparse(r["url"])
                print(f"\n{r['method']} {pu.netloc}{pu.path}  -> {r['status']}")
                if r["auth_headers"]:
                    print(f"  认证头: {list(r['auth_headers'].keys())}")
                print(f"  响应体: {json.dumps(r['body'], ensure_ascii=False)[:500]}")

            print("\n=== 页面按钮文本 ===")
            print(btn_texts)
            print(f"\n[截图已存] {OUT}/zzmeb_panel.png")
        finally:
            await browser.close()


async def main():
    app = get_client(name="my_account", workdir="data")
    async with app:
        auth_url = await get_auth_url(app)
    await probe(auth_url)


if __name__ == "__main__":
    asyncio.run(main())

