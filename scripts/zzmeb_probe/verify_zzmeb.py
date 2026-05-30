"""本地验证 zzmeb_bot 双签配置：动作6(面板签到) 端到端跑通，无副作用。
作者: le.yang
仅 mock 外部边界（Telegram peer / eb.zzm.li HTTP 接口），其余真实执行。
"""
import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

from tg_signer.config import SignChatV3
from tg_signer.core import UserSigner

ZZMEB_BLOCK = {
    "chat_id": 8922873363,
    "name": "娘口三三 (@zzmeb_bot)",
    "delete_after": None,
    "actions": [
        {"action": 1, "text": "/start"},
        {"action": 3, "text": "🎯 签到"},
        {
            "action": 6,
            "bot_username": "zzmeb_bot",
            "api_base_url": None,
            "info_endpoint": "/api/v1/tg/info",
            "checkin_endpoint": "/api/v1/tg/checkin",
            "extra_headers": None,
            "bark_enabled": False,
        },
    ],
    "action_interval": 1.0,
}


class FakeResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.is_success = 200 <= status < 300

    def raise_for_status(self):
        if not self.is_success:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class FakeAsyncClient:
    """模拟 httpx.AsyncClient：拦截 eb.zzm.li 的 info/checkin 接口。"""

    def __init__(self, *args, headers=None, **kwargs):
        self.headers = headers or {}
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, **kwargs):
        self.calls.append(("GET", url))
        # 模拟用户信息：余额63，下次签到时间为过去（即现在可签）
        past = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
        return FakeResp(
            {"message": "Success", "data": {"balance": 63, "next_check_in": past}}
        )

    async def post(self, url, **kwargs):
        self.calls.append(("POST", url))
        # 模拟签到成功，+5 猫币
        return FakeResp({"message": "Success", "data": {"coin": 5}})


def make_signer(tmp):
    return UserSigner(
        task_name="zzmeb-verify",
        account="my_account",
        session_dir=str(tmp),
        workdir=str(tmp / ".signer"),
    )


async def fake_invoke(query):
    """区分 GetFullUser / RequestWebView 两类 raw 调用。"""
    name = type(query).__name__
    if name == "GetFullUser":
        menu_button = SimpleNamespace(url="https://eb.zzm.li/")
        bot_info = SimpleNamespace(menu_button=menu_button)
        return SimpleNamespace(full_user=SimpleNamespace(bot_info=bot_info))
    if name == "RequestWebView":
        # tgWebAppData 放在 URL fragment 中，供 _webview_checkin 解析
        return SimpleNamespace(
            url="https://eb.zzm.li/#tgWebAppData=user%3D%257B%2522id%2522%253A1%257D"
            "%26auth_date%3D1700000000%26hash%3Ddeadbeef&tgWebAppVersion=7.0"
        )
    raise AssertionError(f"unexpected invoke: {name}")


async def main():
    import tempfile
    import pathlib

    results = []

    # 1) 配置解析校验
    chat = SignChatV3.model_validate(ZZMEB_BLOCK)
    descs = [a.action.desc for a in chat.actions]
    results.append(("配置解析", descs == ["发送普通文本", "根据文本点击键盘", "面板页面签到"], descs))

    # 2) 动作6 面板签到端到端（mock 外部边界）
    import tg_signer.core as core

    tmp = pathlib.Path(tempfile.mkdtemp())
    signer = make_signer(tmp)
    signer.app.resolve_peer = AsyncMock(return_value=SimpleNamespace(user_id=1))
    signer.app.invoke = AsyncMock(side_effect=fake_invoke)

    orig_client = core.httpx.AsyncClient
    core.httpx.AsyncClient = FakeAsyncClient
    captured = {}
    try:
        webview_action = chat.actions[2]
        ok = await signer._webview_checkin(webview_action)
        captured["ok"] = ok
    finally:
        core.httpx.AsyncClient = orig_client

    results.append(("面板签到返回成功", captured.get("ok") is True, captured.get("ok")))

    print("\n==== zzmeb 双签本地验证 ====")
    all_ok = True
    for name, passed, detail in results:
        flag = "PASS" if passed else "FAIL"
        all_ok = all_ok and passed
        print(f"[{flag}] {name}: {detail}")
    print("============================")
    print("RESULT:", "ALL PASS" if all_ok else "HAS FAILURE")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))


