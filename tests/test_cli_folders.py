import asyncio

import pytest
from click.testing import CliRunner

import tg_signer.cli.automation as automation_cli
import tg_signer.cli.monitor as monitor_cli
import tg_signer.cli.signer as signer_cli
from tg_signer.core import ChatFolderError


class DummyWorker:
    def __init__(self):
        self.calls = []

    def app_run(self, coroutine=None):
        if coroutine is not None:
            asyncio.run(coroutine)

    async def list_folders(self):
        self.calls.append({"method": "list_folders"})

    async def login(self, num_of_dialogs, folder=None):
        self.calls.append(
            {
                "method": "login",
                "num_of_dialogs": num_of_dialogs,
                "folder": folder,
            }
        )

    async def run(self, num_of_dialogs, folder=None):
        self.calls.append(
            {
                "method": "run",
                "num_of_dialogs": num_of_dialogs,
                "folder": folder,
            }
        )

    async def run_once(self, num_of_dialogs, folder=None):
        self.calls.append(
            {
                "method": "run_once",
                "num_of_dialogs": num_of_dialogs,
                "folder": folder,
            }
        )


@pytest.fixture
def runner():
    return CliRunner()


def test_list_folders_command(monkeypatch, runner):
    worker = DummyWorker()
    monkeypatch.setattr(signer_cli, "get_signer", lambda *_args, **_kwargs: worker)

    result = runner.invoke(signer_cli.tg_signer, ["list-folders"])

    assert result.exit_code == 0
    assert worker.calls == [{"method": "list_folders"}]


@pytest.mark.parametrize(
    ("args", "method"),
    [
        (["login", "--from-folder", "Sign"], "login"),
        (["run", "--from-folder", "Sign", "task"], "run"),
        (["run-once", "--from-folder", "Sign", "task"], "run_once"),
        (
            [
                "multi-run",
                "--account",
                "account-a",
                "--from-folder",
                "Sign",
                "task",
            ],
            "run",
        ),
    ],
)
def test_signer_commands_forward_folder(monkeypatch, runner, args, method):
    worker = DummyWorker()
    monkeypatch.setattr(signer_cli, "get_signer", lambda *_args, **_kwargs: worker)

    result = runner.invoke(signer_cli.tg_signer, args)

    assert result.exit_code == 0, result.output
    assert worker.calls[0]["method"] == method
    assert worker.calls[0]["folder"] == "Sign"


@pytest.mark.parametrize(
    ("args", "module", "factory_name"),
    [
        (
            ["automation", "run", "--from-folder", "Sign", "task"],
            automation_cli,
            "get_automation",
        ),
        (
            ["monitor", "run", "--from-folder", "Sign", "task"],
            monitor_cli,
            "get_monitor",
        ),
    ],
)
def test_subsystem_run_commands_forward_folder(
    monkeypatch, runner, args, module, factory_name
):
    worker = DummyWorker()
    monkeypatch.setattr(module, factory_name, lambda *_args, **_kwargs: worker)

    result = runner.invoke(signer_cli.tg_signer, args)

    assert result.exit_code == 0, result.output
    assert worker.calls == [{"method": "run", "num_of_dialogs": 20, "folder": "Sign"}]


def test_folder_errors_are_reported_as_click_errors(monkeypatch, runner):
    class ErrorWorker(DummyWorker):
        async def login(self, num_of_dialogs, folder=None):
            del num_of_dialogs, folder
            raise ChatFolderError("folder error")

    worker = ErrorWorker()
    monkeypatch.setattr(signer_cli, "get_signer", lambda *_args, **_kwargs: worker)

    result = runner.invoke(
        signer_cli.tg_signer,
        ["login", "--from-folder", "Missing"],
    )

    assert result.exit_code == 1
    assert "Error: folder error" in result.output
