"""探测 zzmeb v4：注入 initData → 点「Telegram一键登入」换 session → 点签到。
抓全程 login/checkin 接口的路径、认证方式、请求体、响应体。
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
    uf = await app.invoke(GetFullUser(id=bot_peer))
    base = uf.full_user.bot_info.menu_button.url
    auth = await app.invoke(
        RequestWebView(peer=bot_peer, bot=bot_peer, platform="ios", url=base)
    )
    return auth.url


def extract(auth_url):
    frag = urlparse(auth_url).fragment
    init_data = parse_qs(frag).get("tgWebAppData", [""])[0]
    unsafe = {k: v[0] for k, v in parse_qs(init_data).items()}
    if "user" in unsafe:
        try:
            unsafe["user"] = json.loads(unsafe["user"])
        except Exception:
            pass
    return init_data, unsafe


def stub(init_data, unsafe):
    payload = {
        "query_id": unsafe.get("query_id"),
        "user": unsafe.get("user") or {},
        "auth_date": unsafe.get("auth_date"),
        "signature": unsafe.get("signature"),
        "hash": unsafe.get("hash"),
    }
    return (
        "window.Telegram=window.Telegram||{};"
        "window.Telegram.WebApp=Object.assign({"
        "ready(){},expand(){},close(){},enableClosingConfirmation(){},"
        "disableClosingConfirmation(){},onEvent(){},offEvent(){},sendData(){},"
        "setHeaderColor(){},setBackgroundColor(){},openLink(){},openTelegramLink(){},"
        "showPopup(){},showAlert(){},showConfirm(){},requestWriteAccess(){},requestContact(){},"
        "HapticFeedback:{impactOccurred(){},notificationOccurred(){},selectionChanged(){}},"
        "MainButton:{show(){},hide(){},setText(){},onClick(){},offClick(){},enable(){},disable(){},setParams(){}},"
        "BackButton:{show(){},hide(){},onClick(){},offClick(){}},"
        "CloudStorage:{getItem(){},setItem(){},getKeys(){}},"
        "colorScheme:'light',themeParams:{},version:'7.0',platform:'ios',"
        "isExpanded:true,viewportHeight:800,viewportStableHeight:800,isClosingConfirmationEnabled:false,"
        f"initData:{json.dumps(init_data)},"
        f"initDataUnsafe:{json.dumps(payload, ensure_ascii=False)}"
        "},window.Telegram.WebApp||{});"
    )


async def probe(auth_url, init_data, unsafe):
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
            await ctx.add_init_script(stub(init_data, unsafe))
            page = await ctx.new_page()
            pending = {}

            def on_request(req):
                if req.resource_type in ("xhr", "fetch"):
                    pending[req] = (dict(req.headers), req.post_data)

            async def on_response(resp):
                req = resp.request
                if req not in pending:
                    return
                headers, post = pending[req]
                try:
                    body = await resp.json()
                except Exception:
                    try:
                        body = (await resp.text())[:200]
                    except Exception:
                        body = None
                records.append({
                    "method": req.method,
                    "path": urlparse(resp.url).path,
                    "auth": {k: (v[:30] + "…" if len(v) > 30 else v)
                             for k, v in headers.items()
                             if k.lower() in ("authorization", "x-initdata",
                                              "x-telegram-init-data", "cookie")},
                    "post": post[:150] if post else None,
                    "status": resp.status,
                    "body": body,
                })

            page.on("request", on_request)
            page.on("response", lambda r: asyncio.create_task(on_response(r)))

            await page.goto(auth_url, wait_until="domcontentloaded", timeout=45000)
            await page.wait_for_timeout(4000)

            async def dump(tag, frm):
                print(f"\n===== {tag} =====")
                for r in records[frm:]:
                    print(f"{r['method']} {r['path']} -> {r['status']}"
                          f"{'  auth='+str(list(r['auth'])) if r['auth'] else ''}")
                    if r["post"]:
                        print(f"   req: {r['post']}")
                    print(f"   resp: {json.dumps(r['body'], ensure_ascii=False)[:300]}")

            await dump("加载阶段", 0)

            # 1) 点登录
            m = len(records)
            for label in ("Telegram一键登入", "一键登入", "登入", "登录", "Login"):
                try:
                    b = page.get_by_text(label, exact=False).first
                    await b.wait_for(state="visible", timeout=4000)
                    await b.click(timeout=4000)
                    print(f"\n[点击登录「{label}」]")
                    await page.wait_for_timeout(4000)
                    break
                except Exception:
                    continue
            await dump("登录后", m)

            btns = await page.eval_on_selector_all(
                "button,[role=button],a",
                "els=>els.map(e=>(e.innerText||'').trim()).filter(Boolean).slice(0,40)")
            print(f"\n[登录后页面按钮] {btns}")
            await page.screenshot(path=f"{OUT}/zzmeb_after_login.png", full_page=True)

            # 2) 点签到
            m = len(records)
            for label in ("立即签到", "立即簽到", "签到", "簽到", "每日签到"):
                try:
                    b = page.get_by_text(label, exact=False).first
                    await b.wait_for(state="visible", timeout=4000)
                    await b.click(timeout=4000)
                    print(f"\n[点击签到「{label}」]")
                    await page.wait_for_timeout(4000)
                    break
                except Exception:
                    continue
            await dump("签到后", m)
        finally:
            await browser.close()


async def main():
    app = get_client(name="my_account", workdir="data")
    async with app:
        auth_url = await get_auth_url(app)
    init_data, unsafe = extract(auth_url)
    print(f"[initData 字段] {list(unsafe.keys())}")
    await probe(auth_url, init_data, unsafe)


if __name__ == "__main__":
    asyncio.run(main())

