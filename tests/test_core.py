import asyncio
import pathlib
from types import SimpleNamespace

import pytest
from pyrogram.types import InlineKeyboardButton, InlineKeyboardMarkup, WebAppInfo

from tg_signer.config import ClickKeyboardByTextAction, OpenWebAppByTextAction
from tg_signer.core import (
    BaseUserWorker,
    UserSigner,
    get_client,
)


class TestBaseUserWorker:
    @pytest.mark.asyncio
    async def test(self):
        BaseUserWorker()


def _clear_client_state():
    """Helper to clear module-level client caches between tests."""
    import tg_signer.core as core

    core._CLIENT_INSTANCES.clear()
    core._CLIENT_REFS.clear()
    core._CLIENT_ASYNC_LOCKS.clear()
    core._LOGIN_ASYNC_LOCKS.clear()
    core._LOGIN_USERS.clear()
    core._API_ASYNC_LOCKS.clear()
    core._API_LAST_CALL_AT.clear()


def test_get_client_caching(tmp_path):
    """get_client should return the same instance for the same key and different
    instances for different keys.
    """
    import tg_signer.core as core

    _clear_client_state()

    name = "acct"
    client1 = get_client(name=name, workdir=tmp_path)
    client2 = get_client(name=name, workdir=tmp_path)
    assert client1 is client2

    # different name -> different key -> different instance
    client3 = get_client(name="other", workdir=tmp_path)
    assert client3 is not client1

    key = str(pathlib.Path(tmp_path).joinpath(name).resolve())
    assert key in core._CLIENT_INSTANCES


@pytest.mark.asyncio
async def test_client_context_manager_reference_counting_and_start_stop(
    monkeypatch, tmp_path
):
    """Test that entering/exiting the async context manager updates reference
    counts, calls start only once for nested entries, and calls stop after
    the final exit. We monkeypatch start/stop to avoid network operations.
    """
    import tg_signer.core as core

    _clear_client_state()

    start_stop_calls = []

    async def fake_start(self):
        # small yield to ensure proper async behavior
        await asyncio.sleep(0)
        start_stop_calls.append("start")
        self._fake_started = True

    async def fake_stop(self):
        await asyncio.sleep(0)
        start_stop_calls.append("stop")
        self._fake_started = False

    monkeypatch.setattr(core.Client, "start", fake_start)
    monkeypatch.setattr(core.Client, "stop", fake_stop)

    name = "acct"
    client = get_client(
        name=name,
        workdir=tmp_path,
    )
    key = client.key
    assert len(core._CLIENT_INSTANCES) == 1
    assert key in core._CLIENT_INSTANCES

    # enter outer context
    async with client as c1:
        assert c1 is client
        # refcount should be 1
        assert core._CLIENT_REFS[key] == 1
        assert getattr(client, "_fake_started", False) is True

        # nested enter should not call start again
        async with client as c2:
            assert c2 is client
            assert core._CLIENT_REFS[key] == 2
            assert getattr(client, "_fake_started", False) is True

        # after inner exit refcount back to 1 and still started
        assert core._CLIENT_REFS[key] == 1
        assert getattr(client, "_fake_started", False) is True

    # after outer exit refcount should be 0 and stop should have been called
    assert core._CLIENT_REFS[key] == 0
    assert getattr(client, "_fake_started", False) is False

    # ensure start and stop each called exactly once
    assert start_stop_calls.count("start") == 1
    assert start_stop_calls.count("stop") == 1

    # instance should be removed from cache after stop
    assert key not in core._CLIENT_INSTANCES


@pytest.mark.asyncio
async def test_login_bootstrap_is_shared_between_concurrent_workers(
    monkeypatch, tmp_path
):
    """Concurrent workers with the same account should only perform one
    get_me/get_dialogs login bootstrap.
    """
    import tg_signer.core as core

    _clear_client_state()
    calls = {"get_me": 0, "get_dialogs": 0, "save_session_string": 0}

    async def fake_start(self):
        await asyncio.sleep(0)

    async def fake_stop(self):
        await asyncio.sleep(0)

    async def fake_get_me(self):
        calls["get_me"] += 1
        await asyncio.sleep(0)
        return SimpleNamespace(id=123456)

    async def fake_get_dialogs(self, limit):
        del limit
        calls["get_dialogs"] += 1
        chat = SimpleNamespace(
            id=10001,
            title="test-chat",
            type="private",
            username=None,
            first_name="test",
            last_name=None,
        )
        yield SimpleNamespace(chat=chat)

    async def fake_save_session_string(self):
        calls["save_session_string"] += 1
        await asyncio.sleep(0)

    monkeypatch.setattr(core.Client, "start", fake_start)
    monkeypatch.setattr(core.Client, "stop", fake_stop)
    monkeypatch.setattr(core.Client, "get_me", fake_get_me)
    monkeypatch.setattr(core.Client, "get_dialogs", fake_get_dialogs)
    monkeypatch.setattr(core.Client, "save_session_string", fake_save_session_string)

    signer1 = UserSigner(
        task_name="task_a",
        account="acct",
        session_dir=tmp_path,
        workdir=tmp_path / ".signer",
    )
    signer2 = UserSigner(
        task_name="task_b",
        account="acct",
        session_dir=tmp_path,
        workdir=tmp_path / ".signer",
    )

    await asyncio.gather(
        signer1.login(num_of_dialogs=20, print_chat=False),
        signer2.login(num_of_dialogs=20, print_chat=False),
    )

    assert calls["get_me"] == 1
    assert calls["get_dialogs"] == 1
    assert calls["save_session_string"] == 1
    assert signer1.user.id == signer2.user.id == 123456


@pytest.mark.asyncio
async def test_call_telegram_api_retries_floodwait(monkeypatch, tmp_path):
    import tg_signer.core as core

    _clear_client_state()
    monkeypatch.setattr(core, "_API_MIN_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr(core, "_API_FLOODWAIT_PADDING_SECONDS", 0.0)
    monkeypatch.setattr(core, "_API_MAX_FLOODWAIT_RETRIES", 2)

    waits = []
    real_sleep = core.asyncio.sleep

    async def fake_sleep(seconds):
        waits.append(seconds)
        await real_sleep(0)

    monkeypatch.setattr(core.asyncio, "sleep", fake_sleep)

    signer = UserSigner(
        task_name="task",
        account="acct",
        session_dir=tmp_path,
        workdir=tmp_path / ".signer",
    )

    called = 0

    async def flaky_api():
        nonlocal called
        called += 1
        if called == 1:
            raise core.errors.FloodWait(2)
        return "ok"

    result = await signer._call_telegram_api("test", flaky_api)

    assert result == "ok"
    assert called == 2
    assert waits == [2]


@pytest.mark.asyncio
async def test_call_telegram_api_is_serialized_for_same_account(monkeypatch, tmp_path):
    import tg_signer.core as core

    _clear_client_state()
    monkeypatch.setattr(core, "_API_MIN_INTERVAL_SECONDS", 0.0)

    signer1 = UserSigner(
        task_name="task_a",
        account="acct",
        session_dir=tmp_path,
        workdir=tmp_path / ".signer",
    )
    signer2 = UserSigner(
        task_name="task_b",
        account="acct",
        session_dir=tmp_path,
        workdir=tmp_path / ".signer",
    )

    active = 0
    max_active = 0

    async def critical_api():
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0)
        active -= 1
        return "done"

    await asyncio.gather(
        signer1._call_telegram_api("critical", critical_api),
        signer2._call_telegram_api("critical", critical_api),
    )

    assert max_active == 1


@pytest.mark.asyncio
async def test_click_keyboard_by_text_ignores_webapp_button(monkeypatch, tmp_path):
    signer = UserSigner(
        task_name="task",
        account="acct",
        session_dir=tmp_path,
        workdir=tmp_path / ".signer",
    )

    called = False

    async def fake_request_callback_answer(*args, **kwargs):
        nonlocal called
        del args, kwargs
        called = True

    monkeypatch.setattr(signer, "request_callback_answer", fake_request_callback_answer)

    message = SimpleNamespace(
        chat=SimpleNamespace(id=123),
        id=456,
        reply_markup=InlineKeyboardMarkup(
            [[InlineKeyboardButton("🎯 签到", web_app=WebAppInfo(url="https://example.com"))]]
        ),
    )

    ok = await signer._click_keyboard_by_text(
        ClickKeyboardByTextAction(text="🎯 签到"), message
    )

    assert ok is False
    assert called is False


@pytest.mark.asyncio
async def test_open_webapp_by_text_uses_button_and_runs_page_action(
    monkeypatch, tmp_path
):
    signer = UserSigner(
        task_name="task",
        account="acct",
        session_dir=tmp_path,
        workdir=tmp_path / ".signer",
    )

    seen = {}

    async def fake_get_webview_url(message, button):
        seen["message"] = message
        seen["button"] = button
        return "https://example.com/auth"

    async def fake_run_webapp(action, webview_url):
        seen["action"] = action
        seen["webview_url"] = webview_url
        return True

    monkeypatch.setattr(signer, "_get_webview_url_from_button", fake_get_webview_url)
    monkeypatch.setattr(signer, "_run_webapp_page_action", fake_run_webapp)

    message = SimpleNamespace(
        chat=SimpleNamespace(id=123),
        id=456,
        reply_markup=InlineKeyboardMarkup(
            [[InlineKeyboardButton("🎯 签到", web_app=WebAppInfo(url="https://example.com"))]]
        ),
    )
    action = OpenWebAppByTextAction(
        text="🎯 签到",
        page_button_text="验证并签到",
        response_url_contains="/auth/checkin/submit",
        success_text="签到成功",
    )

    ok = await signer._open_webapp_by_text(action, message)

    assert ok is True
    assert seen["message"] is message
    assert seen["button"].text == "🎯 签到"
    assert seen["action"] == action
    assert seen["webview_url"] == "https://example.com/auth"


def test_get_twocaptcha_api_key_prefers_action_value(monkeypatch, tmp_path):
    signer = UserSigner(
        task_name="task",
        account="acct",
        session_dir=tmp_path,
        workdir=tmp_path / ".signer",
    )
    monkeypatch.setenv("TWOCAPTCHA_API_KEY", "env-key")

    action = OpenWebAppByTextAction(
        text="🎯 签到",
        page_button_text="验证并签到",
        two_captcha_api_key="action-key",
    )

    assert signer._get_twocaptcha_api_key(action) == "action-key"


def test_get_twocaptcha_api_key_uses_environment(monkeypatch, tmp_path):
    signer = UserSigner(
        task_name="task",
        account="acct",
        session_dir=tmp_path,
        workdir=tmp_path / ".signer",
    )
    monkeypatch.delenv("TWOCAPTCHA_API_KEY", raising=False)
    monkeypatch.setenv("TWO_CAPTCHA_API_KEY", "fallback-key")

    action = OpenWebAppByTextAction(
        text="🎯 签到",
        page_button_text="验证并签到",
    )

    assert signer._get_twocaptcha_api_key(action) == "fallback-key"


@pytest.mark.asyncio
async def test_solve_twocaptcha_image(monkeypatch, tmp_path):
    signer = UserSigner(
        task_name="task",
        account="acct",
        session_dir=tmp_path,
        workdir=tmp_path / ".signer",
    )
    calls = {"post": None, "poll": 0}

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, data):
            calls["post"] = (url, data)
            return FakeResponse({"status": 1, "request": "captcha-id"})

        async def get(self, url, params):
            calls["poll"] += 1
            if calls["poll"] == 1:
                return FakeResponse({"status": 0, "request": "CAPCHA_NOT_READY"})
            return FakeResponse({"status": 1, "request": "ABCD"})

    async def fake_sleep(_):
        return None

    import tg_signer.core as core

    monkeypatch.setattr(core.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(core.asyncio, "sleep", fake_sleep)

    result = await signer._solve_twocaptcha_image(
        api_key="test-key",
        image_bytes=b"image-bytes",
        timeout_seconds=10,
        poll_interval_seconds=1,
    )

    assert result == "ABCD"
    assert calls["post"][0] == "https://2captcha.com/in.php"
    assert calls["post"][1]["key"] == "test-key"
    assert calls["post"][1]["method"] == "base64"
    assert calls["poll"] == 2
