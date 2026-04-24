import asyncio
import logging
from typing import Optional

import click
from click import Group

from tg_signer.automation import UserAutomation

from .signer import tg_signer


def get_automation(
    task_name, ctx_obj: dict, loop: Optional[asyncio.AbstractEventLoop] = None
):
    automation = UserAutomation(
        task_name=task_name,
        account=ctx_obj["account"],
        proxy=ctx_obj["proxy"],
        session_dir=ctx_obj["session_dir"],
        workdir=ctx_obj["workdir"],
        session_string=ctx_obj["session_string"],
        in_memory=ctx_obj["in_memory"],
        loop=loop,
    )
    return automation


@tg_signer.group(name="automation", help="配置和运行自动化")
@click.pass_context
def tg_automation(ctx: click.Context):
    logger = logging.getLogger("tg-signer")
    if ctx.invoked_subcommand in ["run"]:
        if proxy := ctx.obj.get("proxy"):
            logger.info(
                "Using proxy: %s"
                % f"{proxy['scheme']}://{proxy['hostname']}:{proxy['port']}"
            )
        logger.info(f"Using account: {ctx.obj['account']}")


tg_automation: Group


@tg_automation.command(name="list", help="列出已有配置")
@click.pass_obj
def list_(obj):
    return UserAutomation(workdir=obj["workdir"]).list_()


@tg_automation.command(help="根据配置运行自动化")
@click.argument("task_name", nargs=1, default="my_automation")
@click.option(
    "--num-of-dialogs",
    "-n",
    default=20,
    show_default=True,
    type=int,
    help="获取最近N个对话, 请确保想要触发的对话在最近N个对话内",
)
@click.pass_obj
def run(obj, task_name, num_of_dialogs):
    automation = get_automation(task_name, obj)
    automation.app_run(automation.run(num_of_dialogs))


@tg_automation.command(help="初始化或重置配置")
@click.argument("task_name", nargs=1, default="my_automation")
@click.pass_obj
def init(obj, task_name):
    automation = get_automation(task_name, obj)
    config = automation.template_config()
    automation.write_config(config)
    click.echo(f"已生成模板配置: {automation.config_file}")


@tg_automation.command(help="重新配置（使用模板覆盖）")
@click.argument("task_name", nargs=1, default="my_automation")
@click.pass_obj
def reconfig(obj, task_name):
    automation = get_automation(task_name, obj)
    return automation.reconfig()


@tg_automation.command(help="校验配置")
@click.argument("task_name", nargs=1, default="my_automation")
@click.pass_obj
def validate(obj, task_name):
    automation = get_automation(task_name, obj)
    try:
        automation.load_config()
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(str(exc))
    click.echo("配置校验通过")


@tg_automation.command(
    help="""导出配置，默认为输出到终端。\n\n e.g.\n\n  tg-signer automation export -O config.json mytask\n\n  tg-signer automation export mytask > config.json"""
)
@click.argument("task_name")
@click.option(
    "--file", "-O", "file", type=click.Path(), default=None, help="导出至该文件"
)
@click.pass_obj
def export(obj, task_name: str, file: str = None):
    automation = get_automation(task_name, obj)
    data = automation.export()
    if not file:
        click.echo(data)
    else:
        with click.open_file(file, "w", encoding="utf-8") as fp:
            fp.write(data)


@tg_automation.command(
    name="import",
    help="""导入配置，默认为从终端读取。\n\n e.g.\n\n  tg-signer automation import -I config.json mytask\n\n  cat config.json | tg-signer automation import mytask""",
)
@click.argument("task_name")
@click.option(
    "--file", "-I", "file", type=click.Path(), default=None, help="导入该文件"
)
@click.pass_obj
def import_(obj, task_name: str, file: str = None):
    automation = get_automation(task_name, obj)
    if not file:
        stdin_text = click.get_text_stream("stdin")
        data = stdin_text.read()
    else:
        with click.open_file(file, "r", encoding="utf-8") as fp:
            data = fp.read()
    automation.import_(data)
