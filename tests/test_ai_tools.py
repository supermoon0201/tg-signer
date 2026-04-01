import sys
from types import SimpleNamespace

import pytest
from PIL import Image, UnidentifiedImageError

from tg_signer.ai_tools import AITools


class _FakeCompletions:
    async def create(self, **kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"code":"ABCD","option":0,"reason":"match"}'
                    )
                )
            ]
        )


class _FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_FakeCompletions())


@pytest.mark.asyncio
async def test_recognize_gif_code_import_error_mentions_headless_package(
    monkeypatch,
):
    monkeypatch.setattr("tg_signer.ai_tools.get_openai_client", lambda **_: object())
    monkeypatch.setattr(
        "PIL.Image.open",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(UnidentifiedImageError("bad")),
    )

    original_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "cv2":
            raise ImportError("missing cv2")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    tools = AITools({"api_key": "test-key", "model": "gpt-4o-mini"})

    with pytest.raises(ValueError, match="opencv-python-headless"):
        await tools.recognize_gif_code(b"not-a-real-gif", ["A"], client=_FakeClient())


@pytest.mark.asyncio
async def test_recognize_gif_code_uses_cv2_fallback_when_pil_fails(monkeypatch):
    monkeypatch.setattr("tg_signer.ai_tools.get_openai_client", lambda **_: object())
    monkeypatch.setattr(
        "PIL.Image.open",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(UnidentifiedImageError("bad")),
    )

    class _FakeCapture:
        def __init__(self, _path):
            self._reads = 0

        def isOpened(self):
            return True

        def get(self, _prop):
            return 1

        def set(self, _prop, _value):
            return True

        def read(self):
            if self._reads:
                return False, None
            self._reads += 1
            return True, object()

        def release(self):
            return None

    fake_cv2 = SimpleNamespace(
        VideoCapture=_FakeCapture,
        CAP_PROP_FRAME_COUNT=7,
        CAP_PROP_POS_FRAMES=1,
        COLOR_BGR2RGB=4,
        cvtColor=lambda frame, _code: frame,
    )

    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setitem(sys.modules, "numpy", SimpleNamespace())
    monkeypatch.setattr("PIL.Image.fromarray", lambda _frame: Image.new("RGB", (2, 2)))

    tools = AITools({"api_key": "test-key", "model": "gpt-4o-mini"})
    result = await tools.recognize_gif_code(
        b"not-a-real-gif",
        ["ABCD"],
        client=_FakeClient(),
    )

    assert result == 0
