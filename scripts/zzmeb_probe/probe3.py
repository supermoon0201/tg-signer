"""探测 zzmeb v3：注入真实 Telegram.WebApp(initData)，让 SPA 自动登录，
抓取真正的 login / me / checkin 接口路径、请求体、响应体。
作者: le.yang
"""
import asyncio
import json
import os
from urllib.parse import parse_qs, urlparse

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
    return base_url, auth.url


def extract_init_data(auth_url):
    frag = urlparse(auth_url).fragment
    qs = parse_qs(frag)
    init_data = qs.get("tgWebAppData", [""])[0]  # 已解码一次：query_id=..&user=..&hash=..
    unsafe = {k: v[0] for k, v in parse_qs(init_data).items()}
    if "user" in unsafe:
        try:
            unsafe["user"] = json.loads(unsafe["user"])
        except Exception:
            pass
    return init_data, unsafe


def build_webapp_stub(init_data, unsafe):
    """构造注入脚本：在页面加载前伪造 window.Telegram.WebApp。"""
    user = unsafe.get("user") or {}
    payload = {
        "initData": init_data,
        "initDataUnsafe": {
            "query_id": unsafe.get("query_id"),
            "user": user,
            "auth_date": unsafe.get("auth_date"),
            "hash": unsafe.get("hash"),
        },
    }
    return (
        "window.Telegram = window.Telegram || {};"
        "window.Telegram.WebApp = Object.assign({"
        "  ready(){}, expand(){}, close(){}, "
        "  enableClosingConfirmation(){}, disableClosingConfirmation(){},"
        "  onEvent(){}, offEvent(){}, sendData(){}, setHeaderColor(){},"
        "  setBackgroundColor(){}, openLink(){}, openTelegramLink(){},"
        "  showPopup(){}, showAlert(){}, HapticFeedback:{impactOccurred(){},notificationOccurred(){},selectionChanged(){}},"
        "  MainButton:{show(){},hide(){},setText(){},onClick(){},offClick(){},enable(){},disable(){}},"
        "  BackButton:{show(){},hide(){},onClick(){},offClick(){}},"
        "  colorScheme:'light', themeParams:{}, version:'7.0', platform:'ios',"
        "  isExpanded:true, viewportHeight:800, viewportStableHeight:800,"
        f"  initData: {json.dumps(init_data)},"
        f"  initDataUnsafe: {json.dumps(payload['initDataUnsafe'], ensure_ascii=False)}"
        "}, window.Telegram.WebApp || {});"
    )


async def probe(base_url, auth_url, init_data, unsafe):
    from playwright.async_api import async_playwright

    records = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            ctx = await browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) "
                    "AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148"
                )
            )
            await ctx.add_init_script(build_webapp_stub(init_data, unsafe))
            page = await ctx.new_page()
            pending = {}

            def on_request(req):
                if req.resource_type in ("xhr", "fetch"):
                    pending[req] = (dict(req.headers), req.post_data)

            async def on_response(resp):
                req = resp.request
                if req not in pending:
                    return
                headers, post_data = pending[req]
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
                        "auth": {
                            k: (v[:40] + "..." if len(v) > 40 else v)
                            for k, v in headers.items()
                            if k.lower()
                            in ("authorization", "x-initdata", "x-telegram-init-data",
                                "x-init-data", "cookie", "x-sid", "sid")
                        },
                        "post": post_data[:200] if post_data else None,
                        "status": resp.status,
                        "body": body,
                    }
                )

            page.on("request", on_request)
            page.on("response", lambda r: asyncio.create_task(on_response(r)))

            await page.goto(auth_url, wait_until="domcontentloaded", timeout=45000)
            await page.wait_for_timeout(6000)

            btns = await page.eval_on_selector_all(
                "button, [role=button], a",
                "els => els.map(e=>(e.innerText||'').trim()).filter(Boolean).slice(0,40)",
            )
            await page.screenshot(path=f"{OUT}/zzmeb_panel3.png", full_page=True)

            # 尝试点击签到
            before = len(records)
            for label in ("立即签到", "立即簽到", "签到", "簽到"):
                try:
                    b = page.get_by_text(label, exact=False).first
                    await b.wait_for(state="visible", timeout=4000)
                    await b.click(timeout=4000)
                    print(f"[已点击「{label}」]")
                    await page.wait_for_timeout(3500)
                    break
                except Exception:
                    continue

            print("\n=== 所有 API 请求/响应 ===")
            for i, r in enumerate(records):
                pu = urlparse(r["url"])
                tag = "  <<< 点击签到后" if i >= before else ""
                print(f"\n{r['method']} {pu.path}  -> {r['status']}{tag}")
                if r["auth"]:
                    print(f"  认证: {json.dumps(r['auth'], ensure_ascii=False)}")
                if r["post"]:
                    print(f"  请求体: {r['post']}")
                print(f"  响应: {json.dumps(r['body'], ensure_ascii=False)[:400]}")

            print(f"\n=== 页面按钮 ===\n{btns}")
            print(f"\n[截图] {OUT}/zzmeb_panel3.png")
        finally:
            await browser.close()


async def main():
    app = get_client(name="my_account", workdir="data")
    async with app:
        base_url, auth_url = await get_auth_url(app)
    init_data, unsafe = extract_init_data(auth_url)
    print(f"[initData 字段] {list(unsafe.keys())}")
    await probe(base_url, auth_url, init_data, unsafe)


if __name__ == "__main__":
    asyncio.run(main())

