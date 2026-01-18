import asyncio
import json
import logging
import os
import pathlib
import random
import time
from collections import Counter, defaultdict
from datetime import datetime
from datetime import time as dt_time
from datetime import timedelta, timezone
from typing import Annotated, BinaryIO, Generic, List, Optional, Type, TypeVar, Union
from urllib import parse

import httpx
from croniter import CroniterBadCronError, croniter
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pyrogram import Client as BaseClient
from pyrogram import errors, filters
from pyrogram.enums import ChatMembersFilter, ChatType
from pyrogram.handlers import EditedMessageHandler, MessageHandler
from pyrogram.methods.utilities.idle import idle
from pyrogram.session import Session
from pyrogram.storage import MemoryStorage
from pyrogram.types import (
    Chat,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    Object,
    User,
)

from tg_signer.config import (
    ActionT,
    BaseJSONConfig,
    ChooseOptionByGifAction,
    ChooseOptionByImageAction,
    ClickKeyboardByTextAction,
    HttpCallback,
    MatchConfig,
    MonitorConfig,
    ReplyByCalculationProblemAction,
    SendDiceAction,
    SendTextAction,
    SignChatV3,
    SignConfigV3,
    SupportAction,
    UDPForward,
    WebViewCheckinAction,
)

from .ai_tools import AITools, OpenAIConfigManager
from .notification.bark import bark_send
from .notification.server_chan import sc_send
from .utils import UserInput, print_to_user

logger = logging.getLogger("tg-signer")

DICE_EMOJIS = ("🎲", "🎯", "🏀", "⚽", "🎳", "🎰")

Session.START_TIMEOUT = 5  # 原始超时时间为2秒，但一些代理访问会超时，所以这里调大一点

OPENAI_USE_PROMPT = "当前任务需要配置大模型，请确保运行前正确设置`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`等环境变量，或通过`tg-signer llm-config`持久化配置。"


def readable_message(message: Message):
    s = "\nMessage: "
    s += f"\n  text: {message.text or ''}"
    if message.photo:
        s += f"\n  图片: [({message.photo.width}x{message.photo.height}) {message.caption}]"
    if message.reply_markup:
        if isinstance(message.reply_markup, InlineKeyboardMarkup):
            s += "\n  InlineKeyboard: "
            for row in message.reply_markup.inline_keyboard:
                s += "\n   "
                for button in row:
                    s += f"{button.text} | "
    return s


def readable_chat(chat: Chat):
    if chat.type == ChatType.BOT:
        type_ = "BOT"
    elif chat.type == ChatType.GROUP:
        type_ = "群组"
    elif chat.type == ChatType.SUPERGROUP:
        type_ = "超级群组"
    elif chat.type == ChatType.CHANNEL:
        type_ = "频道"
    else:
        type_ = "个人"

    none_or_dash = lambda x: x or "-"  # noqa: E731

    return f"id: {chat.id}, username: {none_or_dash(chat.username)}, title: {none_or_dash(chat.title)}, type: {type_}, name: {none_or_dash(chat.first_name)}"


_CLIENT_INSTANCES: dict[str, "Client"] = {}

# reference counts and async locks for shared client lifecycle management
# Keyed by account name. Use asyncio locks to serialize start/stop operations
# so multiple coroutines in the same process can safely share one Client.
_CLIENT_REFS: defaultdict[str, int] = defaultdict(int)
_CLIENT_ASYNC_LOCKS: dict[str, asyncio.Lock] = {}


class Client(BaseClient):
    def __init__(self, name: str, *args, **kwargs):
        key = kwargs.pop("key", None)
        super().__init__(name, *args, **kwargs)
        self.key = key or str(pathlib.Path(self.workdir).joinpath(self.name).resolve())
        if self.in_memory and not self.session_string:
            self.load_session_string()
            self.storage = MemoryStorage(self.name, self.session_string)

    async def __aenter__(self):
        lock = _CLIENT_ASYNC_LOCKS.get(self.key)
        if lock is None:
            lock = asyncio.Lock()
            _CLIENT_ASYNC_LOCKS[self.key] = lock
        async with lock:
            _CLIENT_REFS[self.key] += 1
            if _CLIENT_REFS[self.key] == 1:
                try:
                    await self.start()
                except ConnectionError:
                    pass
            return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        lock = _CLIENT_ASYNC_LOCKS.get(self.key)
        if lock is None:
            return
        async with lock:
            _CLIENT_REFS[self.key] -= 1
            if _CLIENT_REFS[self.key] == 0:
                try:
                    await self.stop()
                except ConnectionError:
                    pass
                _CLIENT_INSTANCES.pop(self.key, None)

    @property
    def session_string_file(self):
        return self.workdir / (self.name + ".session_string")

    async def save_session_string(self):
        with open(self.session_string_file, "w") as fp:
            fp.write(await self.export_session_string())

    def load_session_string(self):
        logger.info("Loading session_string from local file.")
        if self.session_string_file.is_file():
            with open(self.session_string_file, "r") as fp:
                self.session_string = fp.read()
                logger.info("The session_string has been loaded.")
        return self.session_string

    async def log_out(
        self,
    ):
        await super().log_out()
        if self.session_string_file.is_file():
            os.remove(self.session_string_file)


def get_api_config():
    api_id = int(os.environ.get("TG_API_ID", 611335))
    api_hash = os.environ.get("TG_API_HASH", "d524b414d21f4d37f08684c1df41ac9c")
    return api_id, api_hash


def get_proxy(proxy: str = None):
    proxy = proxy or os.environ.get("TG_PROXY")
    if proxy:
        r = parse.urlparse(proxy)
        return {
            "scheme": r.scheme,
            "hostname": r.hostname,
            "port": r.port,
            "username": r.username,
            "password": r.password,
        }
    return None


def get_client(
    name: str = "my_account",
    proxy: dict = None,
    workdir: Union[str, pathlib.Path] = ".",
    session_string: str = None,
    in_memory: bool = False,
    **kwargs,
) -> Client:
    proxy = proxy or get_proxy()
    api_id, api_hash = get_api_config()
    key = str(pathlib.Path(workdir).joinpath(name).resolve())
    if key in _CLIENT_INSTANCES:
        return _CLIENT_INSTANCES[key]
    client = Client(
        name,
        api_id=api_id,
        api_hash=api_hash,
        proxy=proxy,
        workdir=workdir,
        session_string=session_string,
        in_memory=in_memory,
        key=key,
        **kwargs,
    )
    _CLIENT_INSTANCES[key] = client
    return client


def get_now():
    return datetime.now(tz=timezone(timedelta(hours=8)))


def make_dirs(path: pathlib.Path, exist_ok=True):
    path = pathlib.Path(path)
    if not path.is_dir():
        os.makedirs(path, exist_ok=exist_ok)
    return path


ConfigT = TypeVar("ConfigT", bound=BaseJSONConfig)


class BaseUserWorker(Generic[ConfigT]):
    _workdir = "."
    _tasks_dir = "tasks"
    cfg_cls: Type["ConfigT"] = BaseJSONConfig

    def __init__(
        self,
        task_name: str = None,
        session_dir: str = ".",
        account: str = "my_account",
        proxy=None,
        workdir=None,
        session_string: str = None,
        in_memory: bool = False,
        *,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        self.task_name = task_name or "my_task"
        self._session_dir = pathlib.Path(session_dir)
        self._account = account
        self._proxy = proxy
        if workdir:
            self._workdir = pathlib.Path(workdir)
        self.app = get_client(
            account,
            proxy,
            workdir=self._session_dir,
            session_string=session_string,
            in_memory=in_memory,
            loop=loop,
        )
        self.loop = self.app.loop
        self.user: Optional[User] = None
        self._config = None
        self.context = self.ensure_ctx()

    def ensure_ctx(self):
        return {}

    def app_run(self, coroutine=None):
        if coroutine is not None:
            run = self.loop.run_until_complete
            run(coroutine)
        else:
            self.app.run()

    @property
    def workdir(self) -> pathlib.Path:
        workdir = self._workdir
        make_dirs(workdir)
        return pathlib.Path(workdir)

    @property
    def tasks_dir(self):
        tasks_dir = self.workdir / self._tasks_dir
        make_dirs(tasks_dir)
        return pathlib.Path(tasks_dir)

    @property
    def task_dir(self):
        task_dir = self.tasks_dir / self.task_name
        make_dirs(task_dir)
        return task_dir

    def get_user_dir(self, user: User):
        user_dir = self.workdir / "users" / str(user.id)
        make_dirs(user_dir)
        return user_dir

    @property
    def config_file(self):
        return self.task_dir.joinpath("config.json")

    @property
    def config(self) -> ConfigT:
        return self._config or self.load_config()

    @config.setter
    def config(self, value):
        self._config = value

    def log(self, msg, level: str = "INFO", **kwargs):
        msg = f"账户「{self._account}」- 任务「{self.task_name}」: {msg}"
        if level.upper() == "INFO":
            logger.info(msg, **kwargs)
        elif level.upper() == "WARNING":
            logger.warning(msg, **kwargs)
        elif level.upper() == "ERROR":
            logger.error(msg, **kwargs)
        elif level.upper() == "CRITICAL":
            logger.critical(msg, **kwargs)
        else:
            logger.debug(msg, **kwargs)

    def ask_for_config(self):
        raise NotImplementedError

    def write_config(self, config: BaseJSONConfig):
        with open(self.config_file, "w", encoding="utf-8") as fp:
            json.dump(config.to_jsonable(), fp, ensure_ascii=False)

    def reconfig(self):
        config = self.ask_for_config()
        self.write_config(config)
        return config

    def load_config(self, cfg_cls: Type[ConfigT] = None) -> ConfigT:
        cfg_cls = cfg_cls or self.cfg_cls
        if not self.config_file.exists():
            config = self.reconfig()
        else:
            with open(self.config_file, "r", encoding="utf-8") as fp:
                result = cfg_cls.load(json.load(fp))
                if result is None:
                    raise ValueError(
                        f"配置文件 {self.config_file} 格式不正确或版本不匹配。"
                        f"请检查配置文件格式是否符合要求。"
                    )
                config, from_old = result
                if from_old:
                    self.write_config(config)
        self.config = config
        return config

    def get_task_list(self):
        signs = []
        for d in os.listdir(self.tasks_dir):
            if self.tasks_dir.joinpath(d).is_dir():
                signs.append(d)
        return signs

    def list_(self):
        print_to_user("已配置的任务：")
        for d in self.get_task_list():
            print_to_user(d)

    def set_me(self, user: User):
        self.user = user
        with open(
            self.get_user_dir(user).joinpath("me.json"), "w", encoding="utf-8"
        ) as fp:
            fp.write(str(user))

    async def login(self, num_of_dialogs=20, print_chat=True):
        self.log("开始登录...")
        app = self.app
        async with app:
            me = await app.get_me()
            self.set_me(me)
            latest_chats = []
            async for dialog in app.get_dialogs(num_of_dialogs):
                chat = dialog.chat
                latest_chats.append(
                    {
                        "id": chat.id,
                        "title": chat.title,
                        "type": chat.type,
                        "username": chat.username,
                        "first_name": chat.first_name,
                        "last_name": chat.last_name,
                    }
                )
                if print_chat:
                    print_to_user(readable_chat(chat))

            with open(
                self.get_user_dir(me).joinpath("latest_chats.json"),
                "w",
                encoding="utf-8",
            ) as fp:
                json.dump(
                    latest_chats,
                    fp,
                    indent=4,
                    default=Object.default,
                    ensure_ascii=False,
                )
            await self.app.save_session_string()

    async def logout(self):
        self.log("开始登出...")
        is_authorized = await self.app.connect()
        if not is_authorized:
            await self.app.storage.delete()
            return None
        return await self.app.log_out()

    async def send_message(
        self, chat_id: Union[int, str], text: str, delete_after: int = None, **kwargs
    ):
        """
        发送文本消息
        :param chat_id:
        :param text:
        :param delete_after: 秒, 发送消息后进行删除，``None`` 表示不删除, ``0`` 表示立即删除.
        :param kwargs:
        :return:
        """
        message = await self.app.send_message(chat_id, text, **kwargs)
        if delete_after is not None:
            self.log(
                f"Message「{text}」 to {chat_id} will be deleted after {delete_after} seconds."
            )
            self.log("Waiting...")
            await asyncio.sleep(delete_after)
            await message.delete()
            self.log(f"Message「{text}」 to {chat_id} deleted!")
        return message

    async def send_dice(
        self,
        chat_id: Union[int, str],
        emoji: str = "🎲",
        delete_after: int = None,
        **kwargs,
    ):
        """
        发送DICE类型消息
        :param chat_id:
        :param emoji: Should be one of "🎲", "🎯", "🏀", "⚽", "🎳", or "🎰".
        :param delete_after:
        :param kwargs:
        :return:
        """
        emoji = emoji.strip()
        if emoji not in DICE_EMOJIS:
            self.log(
                f"Warning, emoji should be one of {', '.join(DICE_EMOJIS)}",
                level="WARNING",
            )
        message = await self.app.send_dice(chat_id, emoji, **kwargs)
        if message and delete_after is not None:
            self.log(
                f"Dice「{emoji}」 to {chat_id} will be deleted after {delete_after} seconds."
            )
            self.log("Waiting...")
            await asyncio.sleep(delete_after)
            await message.delete()
            self.log(f"Dice「{emoji}」 to {chat_id} deleted!")
        return message

    async def search_members(
        self, chat_id: Union[int, str], query: str, admin=False, limit=10
    ):
        filter_ = ChatMembersFilter.SEARCH
        if admin:
            filter_ = ChatMembersFilter.ADMINISTRATORS
            query = ""
        async for member in self.app.get_chat_members(
            chat_id, query, limit=limit, filter=filter_
        ):
            yield member

    async def list_members(
        self, chat_id: Union[int, str], query: str = "", admin=False, limit=10
    ):
        async with self.app:
            async for member in self.search_members(chat_id, query, admin, limit):
                print_to_user(
                    User(
                        id=member.user.id,
                        username=member.user.username,
                        first_name=member.user.first_name,
                        last_name=member.user.last_name,
                        is_bot=member.user.is_bot,
                    )
                )

    def export(self):
        with open(self.config_file, "r", encoding="utf-8") as fp:
            data = fp.read()
        return data

    def import_(self, config_str: str):
        with open(self.config_file, "w", encoding="utf-8") as fp:
            fp.write(config_str)

    def ask_one(self):
        raise NotImplementedError

    def ensure_ai_cfg(self):
        cfg_manager = OpenAIConfigManager(self.workdir)
        cfg = cfg_manager.load_config()
        if not cfg:
            cfg = cfg_manager.ask_for_config()
        return cfg

    def get_ai_tools(self):
        return AITools(self.ensure_ai_cfg())


class Waiter:
    def __init__(self):
        self.waiting_ids = set()
        self.waiting_counter = Counter()

    def add(self, elm):
        self.waiting_ids.add(elm)
        self.waiting_counter[elm] += 1

    def discard(self, elm):
        self.waiting_ids.discard(elm)
        self.waiting_counter.pop(elm, None)

    def sub(self, elm):
        self.waiting_counter[elm] -= 1
        if self.waiting_counter[elm] <= 0:
            self.discard(elm)

    def clear(self):
        self.waiting_ids.clear()
        self.waiting_counter.clear()

    def __bool__(self):
        return bool(self.waiting_ids)

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.waiting_counter}>"


class UserSignerWorkerContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    waiter: Waiter
    sign_chats: defaultdict[int, list[SignChatV3]]  # 签到配置列表
    chat_messages: defaultdict[
        int,
        Annotated[
            dict[int, Optional[Message]],
            Field(default_factory=dict),
        ],
    ]  # 收到的消息，key为chat id
    waiting_message: Optional[Message]  # 正在处理的消息


class UserSigner(BaseUserWorker[SignConfigV3]):
    _workdir = ".signer"
    _tasks_dir = "signs"
    cfg_cls = SignConfigV3
    context: UserSignerWorkerContext

    def ensure_ctx(self) -> UserSignerWorkerContext:
        return UserSignerWorkerContext(
            waiter=Waiter(),
            sign_chats=defaultdict(list),
            chat_messages=defaultdict(dict),
            waiting_message=None,
        )

    @property
    def sign_record_file(self):
        sign_record_dir = self.task_dir / str(self.user.id)
        make_dirs(sign_record_dir)
        return sign_record_dir / "sign_record.json"

    def _ask_actions(
        self, input_: UserInput, available_actions: List[SupportAction] = None
    ) -> List[ActionT]:
        print_to_user(f"{input_.index_str}开始配置<动作>，请按照实际签到顺序配置。")
        available_actions = available_actions or list(SupportAction)
        actions = []
        while True:
            try:
                local_input_ = UserInput()
                print_to_user(f"第{len(actions) + 1}个动作: ")
                for action in available_actions:
                    print_to_user(f"  {action.value}: {action.desc}")
                print_to_user()
                action_str = local_input_("输入对应的数字选择动作: ").strip()
                action = SupportAction(int(action_str))
                if action not in available_actions:
                    raise ValueError(f"不支持的动作: {action}")
                if len(actions) == 0 and action not in [
                    SupportAction.SEND_TEXT,
                    SupportAction.SEND_DICE,
                ]:
                    raise ValueError(
                        f"第一个动作必须为「{SupportAction.SEND_TEXT.desc}」或「{SupportAction.SEND_DICE.desc}」"
                    )
                if action == SupportAction.SEND_TEXT:
                    text = local_input_("输入要发送的文本: ")
                    actions.append(SendTextAction(text=text))
                elif action == SupportAction.SEND_DICE:
                    dice = local_input_("输入要发送的骰子（如 🎲, 🎯）: ")
                    actions.append(SendDiceAction(dice=dice))
                elif action == SupportAction.CLICK_KEYBOARD_BY_TEXT:
                    text_of_btn_to_click = local_input_("键盘中需要点击的按钮文本: ")
                    actions.append(ClickKeyboardByTextAction(text=text_of_btn_to_click))
                elif action == SupportAction.CHOOSE_OPTION_BY_IMAGE:
                    print_to_user(
                        "图片识别将使用大模型回答，请确保大模型支持图片识别。"
                    )
                    actions.append(ChooseOptionByImageAction())
                elif action == SupportAction.REPLY_BY_CALCULATION_PROBLEM:
                    print_to_user("计算题将使用大模型回答。")
                    actions.append(ReplyByCalculationProblemAction())
                elif action == SupportAction.CHOOSE_OPTION_BY_GIF:
                    print_to_user(
                        "GIF验证码识别将使用大模型回答。\n"
                        "此动作用于处理：Bot先发送带按钮的消息，再发送GIF验证码的场景。"
                    )
                    actions.append(ChooseOptionByGifAction())
                else:
                    raise ValueError(f"不支持的动作: {action}")
                if local_input_("是否继续添加动作？(y/N)：").strip().lower() != "y":
                    break
            except (ValueError, ValidationError) as e:
                print_to_user("错误: ")
                print_to_user(e)
        input_.incr()
        return actions

    def ask_one(self) -> SignChatV3:
        input_ = UserInput(numbering_lang="chinese_simple")
        chat_id = int(input_("Chat ID（登录时最近对话输出中的ID）: "))
        name = input_("Chat名称（可选）: ")
        actions = self._ask_actions(input_)
        delete_after = (
            input_(
                "等待N秒后删除消息（发送消息后等待进行删除, '0'表示立即删除, 不需要删除直接回车）, N: "
            )
            or None
        )
        if delete_after:
            delete_after = int(delete_after)
        cfgs = {
            "chat_id": chat_id,
            "name": name,
            "delete_after": delete_after,
            "actions": actions,
        }
        return SignChatV3.model_validate(cfgs)

    def ask_for_config(self) -> "SignConfigV3":
        chats = []
        i = 1
        print_to_user(f"开始配置任务<{self.task_name}>\n")
        while True:
            print_to_user(f"第{i}个任务: ")
            try:
                chat = self.ask_one()
                print_to_user(chat)
                print_to_user(f"第{i}个任务配置成功\n")
                chats.append(chat)
            except Exception as e:
                print_to_user(e)
                print_to_user("配置失败")
                i -= 1
            continue_ = input("继续配置任务？(y/N)：")
            if continue_.strip().lower() != "y":
                break
            i += 1
        sign_at_prompt = "签到时间（time或crontab表达式，如'06:00:00'或'0 6 * * *'）: "
        sign_at_str = input(sign_at_prompt) or "06:00:00"
        while not (sign_at := self._validate_sign_at(sign_at_str)):
            print_to_user("请输入正确的时间格式")
            sign_at_str = input(sign_at_prompt) or "06:00:00"

        random_seconds_str = input("签到时间误差随机秒数（默认为0）: ") or "0"
        random_seconds = int(float(random_seconds_str))
        config = SignConfigV3.model_validate(
            {
                "chats": chats,
                "sign_at": sign_at,
                "random_seconds": random_seconds,
            }
        )
        if config.requires_ai:
            print_to_user(OPENAI_USE_PROMPT)
        return config

    @classmethod
    def _validate_sign_at(cls, sign_at_str: str) -> Optional[str]:
        sign_at_str = sign_at_str.replace("：", ":").strip()

        try:
            sign_at = dt_time.fromisoformat(sign_at_str)
            crontab_expr = cls._time_to_crontab(sign_at)
        except ValueError:
            try:
                croniter(sign_at_str)
                crontab_expr = sign_at_str
            except CroniterBadCronError:
                return None
        return crontab_expr

    @staticmethod
    def _time_to_crontab(sign_at: time) -> str:
        return f"{sign_at.minute} {sign_at.hour} * * *"

    def load_sign_record(self):
        sign_record = {}
        if not self.sign_record_file.is_file():
            with open(self.sign_record_file, "w", encoding="utf-8") as fp:
                json.dump(sign_record, fp)
        else:
            with open(self.sign_record_file, "r", encoding="utf-8") as fp:
                sign_record = json.load(fp)
        return sign_record

    async def sign_a_chat(
        self,
        chat: SignChatV3,
        max_flow_retries: int = 3,
    ):
        self.log(f"开始执行: \n{chat}")
        
        for flow_attempt in range(max_flow_retries):
            if flow_attempt > 0:
                self.log(f"第 {flow_attempt + 1} 次从头重新执行签到流程...")
                await asyncio.sleep(2)
            
            need_retry = False
            for action in chat.actions:
                self.log(f"等待处理动作: {action}")
                result = await self.wait_for(chat, action)
                if result == "RETRY_FLOW":
                    self.log("验证码失败，将从头重新执行签到流程", level="WARNING")
                    need_retry = True
                    break
                self.log(f"处理完成: {action}")
                self.context.waiting_message = None
                await asyncio.sleep(chat.action_interval)
            
            if not need_retry:
                return  # 成功完成
        
        self.log(f"签到流程重试 {max_flow_retries} 次后仍失败", level="ERROR")

    async def run(
        self, num_of_dialogs=20, only_once: bool = False, force_rerun: bool = False
    ):
        if self.app.in_memory or self.app.session_string:
            return await self.in_memory_run(
                num_of_dialogs, only_once=only_once, force_rerun=force_rerun
            )
        return await self.normal_run(
            num_of_dialogs, only_once=only_once, force_rerun=force_rerun
        )

    async def in_memory_run(
        self, num_of_dialogs=20, only_once: bool = False, force_rerun: bool = False
    ):
        async with self.app:
            await self.normal_run(
                num_of_dialogs, only_once=only_once, force_rerun=force_rerun
            )

    async def normal_run(
        self, num_of_dialogs=20, only_once: bool = False, force_rerun: bool = False
    ):
        if self.user is None:
            await self.login(num_of_dialogs, print_chat=True)

        config = self.load_config(self.cfg_cls)
        if config.requires_ai:
            self.ensure_ai_cfg()

        sign_record = self.load_sign_record()
        chat_ids = [c.chat_id for c in config.chats]

        async def sign_once():
            for chat in config.chats:
                self.context.sign_chats[chat.chat_id].append(chat)
                try:
                    await self.sign_a_chat(chat)
                except errors.RPCError as _e:
                    self.log(f"签到失败: {_e} \nchat: \n{chat}")
                    logger.warning(_e, exc_info=True)
                    continue

                self.context.chat_messages[chat.chat_id].clear()
                await asyncio.sleep(config.sign_interval)
            sign_record[str(now.date())] = now.isoformat()
            with open(self.sign_record_file, "w", encoding="utf-8") as fp:
                json.dump(sign_record, fp)

        def need_sign(last_date_str):
            if force_rerun:
                return True
            if last_date_str not in sign_record:
                return True
            _last_sign_at = datetime.fromisoformat(sign_record[last_date_str])
            self.log(f"上次执行时间: {_last_sign_at}")
            _cron_it = croniter(self._validate_sign_at(config.sign_at), _last_sign_at)
            _next_run: datetime = _cron_it.next(datetime)
            if _next_run > now:
                self.log("当前未到下次执行时间，无需执行")
                return False
            return True

        while True:
            self.log(f"为以下Chat添加消息回调处理函数：{chat_ids}")
            self.app.add_handler(
                MessageHandler(self.on_message, filters.chat(chat_ids))
            )
            self.app.add_handler(
                EditedMessageHandler(self.on_edited_message, filters.chat(chat_ids))
            )
            try:
                async with self.app:
                    now = get_now()
                    self.log(f"当前时间: {now}")
                    now_date_str = str(now.date())
                    self.context = self.ensure_ctx()
                    if need_sign(now_date_str):
                        await sign_once()

            except (OSError, errors.Unauthorized) as e:
                logger.exception(e)
                await asyncio.sleep(30)
                continue

            if only_once:
                break
            cron_it = croniter(self._validate_sign_at(config.sign_at), now)
            next_run: datetime = cron_it.next(datetime) + timedelta(
                seconds=random.randint(0, int(config.random_seconds))
            )
            self.log(f"下次运行时间: {next_run}")
            await asyncio.sleep((next_run - now).total_seconds())

    async def run_once(self, num_of_dialogs):
        return await self.run(num_of_dialogs, only_once=True, force_rerun=True)

    async def send_text(
        self, chat_id: int, text: str, delete_after: int = None, **kwargs
    ):
        if self.user is None:
            await self.login(print_chat=False)
        async with self.app:
            await self.send_message(chat_id, text, delete_after, **kwargs)

    async def send_dice_cli(
        self,
        chat_id: Union[str, int],
        emoji: str = "🎲",
        delete_after: int = None,
        **kwargs,
    ):
        if self.user is None:
            await self.login(print_chat=False)
        async with self.app:
            await self.send_dice(chat_id, emoji, delete_after, **kwargs)

    async def _on_message(self, client: Client, message: Message):
        chats = self.context.sign_chats.get(message.chat.id)
        if not chats:
            self.log("忽略意料之外的聊天", level="WARNING")
            return
        self.context.chat_messages[message.chat.id][message.id] = message

    async def on_message(self, client: Client, message: Message):
        self.log(
            f"收到来自「{message.from_user.username or message.from_user.id}」的消息: {readable_message(message)}"
        )
        await self._on_message(client, message)

    async def on_edited_message(self, client, message: Message):
        self.log(
            f"收到来自「{message.from_user.username or message.from_user.id}」对消息的更新，消息: {readable_message(message)}"
        )
        # 避免更新正在处理的消息，等待处理完成
        while (
            self.context.waiting_message
            and self.context.waiting_message.id == message.id
        ):
            await asyncio.sleep(0.3)
        await self._on_message(client, message)

    async def _click_keyboard_by_text(
        self, action: ClickKeyboardByTextAction, message: Message
    ):
        if reply_markup := message.reply_markup:
            if isinstance(reply_markup, InlineKeyboardMarkup):
                flat_buttons = (b for row in reply_markup.inline_keyboard for b in row)
                option_to_btn: dict[str, InlineKeyboardButton] = {}
                for btn in flat_buttons:
                    option_to_btn[btn.text] = btn
                    if action.text in btn.text:
                        self.log(f"点击按钮: {btn.text}")
                        await self.request_callback_answer(
                            self.app,
                            message.chat.id,
                            message.id,
                            btn.callback_data,
                        )
                        return True
        return False

    async def _reply_by_calculation_problem(
        self, action: ReplyByCalculationProblemAction, message
    ):
        if message.text:
            self.log("检测到文本回复，尝试调用大模型进行计算题回答")
            self.log(f"问题: \n{message.text}")
            answer = await self.get_ai_tools().calculate_problem(message.text)
            self.log(f"回答为: {answer}")
            await self.send_message(message.chat.id, answer)
            return True
        return False

    async def _choose_option_by_image(self, action: ChooseOptionByImageAction, message):
        if reply_markup := message.reply_markup:
            if isinstance(reply_markup, InlineKeyboardMarkup) and message.photo:
                flat_buttons = (b for row in reply_markup.inline_keyboard for b in row)
                option_to_btn = {btn.text: btn for btn in flat_buttons if btn.text}
                self.log("检测到图片，尝试调用大模型进行图片识别并选择选项")
                image_buffer: BinaryIO = await self.app.download_media(
                    message.photo.file_id, in_memory=True
                )
                image_buffer.seek(0)
                image_bytes = image_buffer.read()
                options = list(option_to_btn)
                result_index = await self.get_ai_tools().choose_option_by_image(
                    image_bytes,
                    "选择正确的选项",
                    list(enumerate(options)),
                )
                result = options[result_index]
                self.log(f"选择结果为: {result}")
                target_btn = option_to_btn.get(result.strip())
                if not target_btn:
                    self.log("未找到匹配的按钮", level="WARNING")
                    return False
                await self.request_callback_answer(
                    self.app,
                    message.chat.id,
                    message.id,
                    target_btn.callback_data,
                )
                return True
        return False

    async def _choose_option_by_gif(
        self,
        action: ChooseOptionByGifAction,
        button_message: Message,
        gif_message: Message,
        max_retries: int = 3,
    ) -> bool:
        """
        根据GIF验证码选择正确选项

        Args:
            action: ChooseOptionByGifAction动作
            button_message: 包含选项按钮的消息
            gif_message: 包含GIF验证码的消息
            max_retries: 最大重试次数

        Returns:
            是否成功选择选项
        """
        reply_markup = button_message.reply_markup
        if not isinstance(reply_markup, InlineKeyboardMarkup):
            self.log("按钮消息没有InlineKeyboard", level="WARNING")
            return False

        # 获取按钮选项
        flat_buttons = (b for row in reply_markup.inline_keyboard for b in row)
        option_to_btn = {btn.text: btn for btn in flat_buttons if btn.text}

        if not option_to_btn:
            self.log("未找到可选按钮", level="WARNING")
            return False

        # 下载GIF
        self.log("检测到GIF验证码，尝试调用大模型进行识别")

        # GIF可能是animation或document
        if gif_message.animation:
            file_id = gif_message.animation.file_id
        elif gif_message.document:
            file_id = gif_message.document.file_id
        elif gif_message.photo:
            file_id = gif_message.photo.file_id
        else:
            self.log("无法获取GIF文件", level="WARNING")
            return False

        gif_buffer: BinaryIO = await self.app.download_media(file_id, in_memory=True)
        gif_buffer.seek(0)
        gif_bytes = gif_buffer.read()

        options = list(option_to_btn)
        self.log(f"选项列表: {options}")

        # 重试逻辑
        last_error = None
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    self.log(f"第 {attempt + 1} 次尝试识别GIF验证码...")
                
                result_index = await self.get_ai_tools().recognize_gif_code(
                    gif_bytes,
                    options,
                )
                
                if result_index < 0 or result_index >= len(options):
                    self.log(f"AI返回的索引超出范围: {result_index}", level="WARNING")
                    last_error = ValueError(f"索引超出范围: {result_index}")
                    continue

                result = options[result_index]
                self.log(f"GIF验证码识别结果为: {result}")

                target_btn = option_to_btn.get(result.strip())
                if not target_btn:
                    self.log("未找到匹配的按钮", level="WARNING")
                    last_error = ValueError("未找到匹配的按钮")
                    continue

                await self.request_callback_answer(
                    self.app,
                    button_message.chat.id,
                    button_message.id,
                    target_btn.callback_data,
                )
                
                # 等待Bot响应，检查验证结果
                await asyncio.sleep(1.5)  # 等待Bot处理
                
                # 检查是否收到验证码错误的响应
                verification_failed = False
                chat_id = button_message.chat.id
                if chat_id in self.context.chat_messages:
                    for msg in list(self.context.chat_messages[chat_id].values()):
                        if msg is None:
                            continue
                        # 检查消息文本或图片标题中是否包含"验证码错误"
                        msg_text = ""
                        if msg.text:
                            msg_text = msg.text
                        elif msg.caption:
                            msg_text = msg.caption
                        
                        if "验证码错误" in msg_text:
                            verification_failed = True
                            self.log(
                                f"验证码验证失败，Bot返回: {msg_text[:50]}",
                                level="WARNING"
                            )
                            break
                
                if verification_failed:
                    last_error = ValueError("验证码错误")
                    if attempt < max_retries - 1:
                        self.log(f"将进行第 {attempt + 2} 次重试...")
                        await asyncio.sleep(2)
                    continue
                
                self.log(f"GIF验证码识别成功（尝试 {attempt + 1}/{max_retries} 次）")
                return True
                
            except Exception as e:
                last_error = e
                self.log(f"第 {attempt + 1} 次识别失败: {e}", level="WARNING")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)  # 重试前等待2秒
        
        # 所有重试都失败
        self.log(f"GIF验证码识别失败，已重试 {max_retries} 次: {last_error}", level="ERROR")
        return False

    async def _send_bark_notification(
        self, action: WebViewCheckinAction, title: str, body: str
    ) -> None:
        """发送Bark通知的辅助方法"""
        if action.bark_enabled:
            bark_url = os.environ.get("BARK_URL")
            if not bark_url:
                self.log("未配置Bark URL（环境变量BARK_URL）", level="WARNING")
                return

            bark_sound = os.environ.get("BARK_SOUND")
            bark_group = os.environ.get("BARK_GROUP")

            await bark_send(
                bark_url=bark_url,
                title=title,
                body=body,
                sound=bark_sound,
                group=bark_group,
            )

    async def _webview_checkin(self, action: WebViewCheckinAction) -> bool:
        """执行 WebView 面板页面签到"""
        from pyrogram.raw.functions.messages import RequestWebView
        from pyrogram.raw.functions.users import GetFullUser

        try:
            # 1. 获取 bot 的 peer
            bot_peer = await self.app.resolve_peer(action.bot_username)

            # 2. 获取 bot 的完整信息
            user_full = await self.app.invoke(GetFullUser(id=bot_peer))

            # 3. 获取 bot 的菜单按钮 URL
            if (
                not user_full.full_user.bot_info
                or not user_full.full_user.bot_info.menu_button
            ):
                error_msg = "Bot 没有菜单按钮"
                self.log(f"Bot {action.bot_username} {error_msg}", level="WARNING")
                await self._send_bark_notification(
                    action,
                    f"WebView签到失败 - {action.bot_username}",
                    f"签到失败: {error_msg}",
                )
                return False

            url = user_full.full_user.bot_info.menu_button.url

            # 4. 请求 WebView 获取认证 URL
            url_auth = (
                await self.app.invoke(
                    RequestWebView(peer=bot_peer, bot=bot_peer, platform="ios", url=url)
                )
            ).url

            # 5. 从 URL 中提取 tgWebAppData 参数
            from urllib.parse import parse_qs, urlparse

            scheme = urlparse(url_auth)
            params = parse_qs(scheme.fragment)
            webapp_data = params.get("tgWebAppData", [""])[0]

            if not webapp_data:
                error_msg = "无法从 WebView URL 中提取 tgWebAppData"
                self.log(error_msg, level="WARNING")
                await self._send_bark_notification(
                    action,
                    f"WebView签到失败 - {action.bot_username}",
                    f"签到失败: {error_msg}",
                )
                return False

            # 6. 构建 API 基础 URL
            if action.api_base_url:
                base_url = action.api_base_url
            else:
                parsed_url = urlparse(url_auth)
                base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"

            url_info = f"{base_url}{action.info_endpoint}"
            url_checkin = f"{base_url}{action.checkin_endpoint}"

            # 7. 准备请求头
            headers = {"X-Initdata": webapp_data}
            if action.extra_headers:
                headers.update(action.extra_headers)

            # 8. 使用 httpx 发送请求
            async with httpx.AsyncClient(headers=headers, timeout=30.0) as client:
                # 先获取用户信息
                self.log("正在获取用户信息...")
                resp_info = await client.get(url_info)
                info_results = resp_info.json()

                if info_results.get("message") != "Success":
                    error_msg = (
                        f"获取用户信息失败: {info_results.get('message', '未知错误')}"
                    )
                    self.log(error_msg, level="WARNING")
                    await self._send_bark_notification(
                        action,
                        f"WebView签到失败 - {action.bot_username}",
                        f"签到失败: {error_msg}",
                    )
                    return False

                # 获取当前余额
                current_balance = info_results.get("data", {}).get("balance", 0)
                self.log(f"当前余额: {current_balance}")

                # 检查下次签到时间
                next_checkin_str = info_results.get("data", {}).get("next_check_in")
                if next_checkin_str:
                    from datetime import datetime, timezone

                    next_checkin_time = datetime.fromisoformat(
                        next_checkin_str.split(".")[0].replace("Z", "+00:00")
                    ).replace(tzinfo=timezone.utc)

                    if next_checkin_time > datetime.now(timezone.utc):
                        self.log(
                            f"还未到签到时间，下次签到时间: {next_checkin_time}",
                            level="INFO",
                        )
                        # 发送信息通知，让用户知道系统正在运行
                        # 转换为北京时间显示
                        beijing_tz = timezone(timedelta(hours=8))
                        beijing_time = next_checkin_time.astimezone(beijing_tz)
                        await self._send_bark_notification(
                            action,
                            f"WebView签到 - {action.bot_username}",
                            f"未到签到时间\n下次签到: {beijing_time.strftime('%Y-%m-%d %H:%M:%S')} (北京时间)",
                        )
                        return False

                # 执行签到
                self.log("正在执行签到...")
                resp = await client.post(url_checkin)
                results = resp.json()
                message = results.get("message", "")

                if any(s in message for s in ("未找到用户", "权限错误")):
                    error_msg = f"账户错误 - {message}"
                    self.log(f"签到失败: {error_msg}", level="WARNING")
                    await self._send_bark_notification(
                        action,
                        f"WebView签到失败 - {action.bot_username}",
                        f"签到失败: {error_msg}",
                    )
                    return False

                if "Failed" in message:
                    self.log(f"签到失败: {message}", level="WARNING")
                    await self._send_bark_notification(
                        action,
                        f"WebView签到失败 - {action.bot_username}",
                        f"签到失败: {message}",
                    )
                    return False
                elif "Success" in message:
                    coin = results.get("data", {}).get("coin", 0)
                    new_balance = current_balance + coin
                    self.log(f"签到成功: +{coin} 分 -> {new_balance} 分")
                    await self._send_bark_notification(
                        action,
                        f"WebView签到成功 - {action.bot_username}",
                        f"签到成功: +{coin} 分 -> {new_balance} 分",
                    )
                    return True
                else:
                    error_msg = f"接收到异常返回信息: {results}"
                    self.log(error_msg, level="WARNING")
                    await self._send_bark_notification(
                        action,
                        f"WebView签到失败 - {action.bot_username}",
                        f"签到失败: 异常返回信息",
                    )
                    return False

        except Exception as e:
            error_msg = str(e)
            self.log(f"WebView 签到失败: {error_msg}", level="ERROR")
            import traceback

            self.log(traceback.format_exc(), level="DEBUG")
            await self._send_bark_notification(
                action,
                f"WebView签到异常 - {action.bot_username}",
                f"签到异常: {error_msg}",
            )
            return False

    async def wait_for(self, chat: SignChatV3, action: ActionT, timeout=10):
        if isinstance(action, SendTextAction):
            return await self.send_message(chat.chat_id, action.text, chat.delete_after)
        elif isinstance(action, SendDiceAction):
            return await self.send_dice(chat.chat_id, action.dice, chat.delete_after)
        elif isinstance(action, WebViewCheckinAction):
            return await self._webview_checkin(action)

        # 特殊处理GIF验证码场景：需要等待两条消息
        if isinstance(action, ChooseOptionByGifAction):
            return await self._wait_for_gif_action(chat, action, timeout)

        self.context.waiter.add(chat.chat_id)
        start = time.perf_counter()
        last_message = None
        while time.perf_counter() - start < timeout:
            await asyncio.sleep(0.3)
            messages_dict = self.context.chat_messages.get(chat.chat_id)
            if not messages_dict:
                continue
            messages = list(messages_dict.values())
            # 暂无新消息
            if messages[-1] == last_message:
                continue
            last_message = messages[-1]
            for message in messages:
                self.context.waiting_message = message
                ok = False
                if isinstance(action, ClickKeyboardByTextAction):
                    ok = await self._click_keyboard_by_text(action, message)
                elif isinstance(action, ReplyByCalculationProblemAction):
                    ok = await self._reply_by_calculation_problem(action, message)
                elif isinstance(action, ChooseOptionByImageAction):
                    ok = await self._choose_option_by_image(action, message)
                if ok:
                    self.context.waiter.sub(message.chat.id)
                    # 将消息ID对应value置为None，保证收到消息的编辑时消息所处的顺序
                    self.context.chat_messages[chat.chat_id][message.id] = None
                    return None
                self.log(f"忽略消息: {readable_message(message)}")
        self.log(f"等待超时: \nchat: \n{chat} \naction: {action}", level="WARNING")
        return None

    async def _wait_for_gif_action(
        self, chat: SignChatV3, action: ChooseOptionByGifAction, timeout=10
    ):
        """
        处理GIF验证码场景：等待按钮消息和GIF消息，然后执行识别
        验证码失败时返回特殊标记，由上层重新执行整个签到流程
        """
        self.context.waiter.add(chat.chat_id)
        start = time.perf_counter()
        button_message = None
        gif_message = None
        processed_ids = set()

        while time.perf_counter() - start < timeout:
            await asyncio.sleep(0.3)
            messages_dict = self.context.chat_messages.get(chat.chat_id)
            if not messages_dict:
                continue

            messages = [
                m for m in messages_dict.values() if m and m.id not in processed_ids
            ]

            for message in messages:
                processed_ids.add(message.id)
                self.context.waiting_message = message

                # 检查是否是带按钮的消息
                if message.reply_markup and isinstance(
                    message.reply_markup, InlineKeyboardMarkup
                ):
                    button_message = message
                    self.log(f"检测到按钮消息: {readable_message(message)}")

                # 检查是否是GIF/动图消息
                if message.animation or (
                    message.document
                    and message.document.mime_type
                    and "gif" in message.document.mime_type.lower()
                ):
                    gif_message = message
                    self.log(f"检测到GIF消息: {readable_message(message)}")

                # 如果两者都有，执行识别
                if button_message and gif_message:
                    ok = await self._choose_option_by_gif(
                        action, button_message, gif_message
                    )
                    if ok:
                        self.context.waiter.sub(chat.chat_id)
                        self.context.chat_messages[chat.chat_id][button_message.id] = (
                            None
                        )
                        self.context.chat_messages[chat.chat_id][gif_message.id] = None
                        return None
                    else:
                        # 识别失败，返回重试标记
                        self.context.waiter.sub(chat.chat_id)
                        self.context.chat_messages[chat.chat_id].clear()
                        return "RETRY_FLOW"

        self.log(
            f"等待GIF验证码超时: \nchat: \n{chat} \naction: {action}", level="WARNING"
        )
        self.context.waiter.sub(chat.chat_id)
        return None

    async def request_callback_answer(
        self,
        client: Client,
        chat_id: Union[int, str],
        message_id: int,
        callback_data: Union[str, bytes],
        **kwargs,
    ):
        try:
            await client.request_callback_answer(
                chat_id, message_id, callback_data=callback_data, **kwargs
            )
            self.log("点击完成")
        except (errors.BadRequest, TimeoutError) as e:
            self.log(e, level="ERROR")

    async def schedule_messages(
        self,
        chat_id: Union[int, str],
        text: str,
        crontab: str = None,
        next_times: int = 1,
        random_seconds: int = 0,
    ):
        now = get_now()
        it = croniter(crontab, start_time=now)
        if self.user is None:
            await self.login(print_chat=False)
        results = []
        async with self.app:
            for n in range(next_times):
                next_dt: datetime = it.next(ret_type=datetime) + timedelta(
                    seconds=random.randint(0, random_seconds)
                )
                results.append({"at": next_dt.isoformat(), "text": text})
                await self.app.send_message(
                    chat_id,
                    text,
                    schedule_date=next_dt,
                )
                await asyncio.sleep(0.1)
                print_to_user(f"已配置次数：{n + 1}")
        self.log(f"已配置定时发送消息，次数{next_times}")
        return results

    async def get_schedule_messages(self, chat_id):
        if self.user is None:
            await self.login(print_chat=False)
        async with self.app:
            messages = await self.app.get_scheduled_messages(chat_id)
            for message in messages:
                print_to_user(f"{message.date}: {message.text}")


class UserMonitor(BaseUserWorker[MonitorConfig]):
    _workdir = ".monitor"
    _tasks_dir = "monitors"
    cfg_cls = MonitorConfig
    config: MonitorConfig

    def ask_one(self):
        input_ = UserInput()
        chat_id = (input_("Chat ID（登录时最近对话输出中的ID）: ")).strip()
        if not chat_id.startswith("@"):
            chat_id = int(chat_id)
        rules = ["exact", "contains", "regex", "all"]
        while rule := (input_(f"匹配规则({', '.join(rules)}): ") or "exact"):
            if rule in rules:
                break
            print_to_user("不存在的规则, 请重新输入!")
        rule_value = None
        if rule != "all":
            while not (rule_value := input_("规则值（不可为空）: ")):
                print_to_user("不可为空！")
                continue
        from_user_ids = (
            input_(
                "只匹配来自特定用户ID的消息（多个用逗号隔开, 匹配所有用户直接回车）: "
            )
            or None
        )
        always_ignore_me = input_("总是忽略自己发送的消息（y/N）: ").lower() == "y"
        if from_user_ids:
            from_user_ids = [
                i if i.startswith("@") else int(i) for i in from_user_ids.split(",")
            ]
        default_send_text = input_("默认发送文本（不需要则回车）: ") or None
        ai_reply = False
        ai_prompt = None
        use_ai_reply = input_("是否使用AI进行回复(y/N): ") or "n"
        if use_ai_reply.lower() == "y":
            ai_reply = True
            while not (ai_prompt := input_("输入你的提示词（作为`system prompt`）: ")):
                print_to_user("不可为空！")
                continue
            print_to_user(OPENAI_USE_PROMPT)

        send_text_search_regex = None
        if not ai_reply:
            send_text_search_regex = (
                input_("从消息中提取发送文本的正则表达式（不需要则直接回车）: ") or None
            )

        if default_send_text or ai_reply or send_text_search_regex:
            delete_after = (
                input_(
                    "发送消息后等待N秒进行删除（'0'表示立即删除, 不需要删除直接回车）， N: "
                )
                or None
            )
            if delete_after:
                delete_after = int(delete_after)
            forward_to_chat_id = (
                input_("转发消息到该聊天ID，默认为消息来源：")
            ).strip()
            if forward_to_chat_id and not forward_to_chat_id.startswith("@"):
                forward_to_chat_id = int(forward_to_chat_id)
        else:
            delete_after = None
            forward_to_chat_id = None

        push_via_server_chan = (
            input_("是否通过Server酱推送消息(y/N): ") or "n"
        ).lower() == "y"
        server_chan_send_key = None
        if push_via_server_chan:
            server_chan_send_key = (
                input_(
                    "Server酱的SendKey（不填将从环境变量`SERVER_CHAN_SEND_KEY`读取）: "
                )
                or None
            )

        forward_to_external = (
            input_("是否需要转发到外部（UDP, Http）(y/N): ").lower() == "y"
        )
        external_forwards = None
        if forward_to_external:
            external_forwards = []
            if input_("是否需要转发到UDP(y/N): ").lower() == "y":
                addr = input_("请输入UDP服务器地址和端口（形如`127.0.0.1:1234`）: ")
                host, port = addr.split(":")
                external_forwards.append(
                    {
                        "host": host,
                        "port": int(port),
                    }
                )

            if input_("是否需要转发到Http(y/N): ").lower() == "y":
                url = input_("请输入Http地址（形如`http://127.0.0.1:1234`）: ")
                external_forwards.append(
                    {
                        "url": url,
                    }
                )

        return MatchConfig.model_validate(
            {
                "chat_id": chat_id,
                "rule": rule,
                "rule_value": rule_value,
                "from_user_ids": from_user_ids,
                "always_ignore_me": always_ignore_me,
                "default_send_text": default_send_text,
                "ai_reply": ai_reply,
                "ai_prompt": ai_prompt,
                "send_text_search_regex": send_text_search_regex,
                "delete_after": delete_after,
                "forward_to_chat_id": forward_to_chat_id,
                "push_via_server_chan": push_via_server_chan,
                "server_chan_send_key": server_chan_send_key,
                "external_forwards": external_forwards,
            }
        )

    def ask_for_config(self) -> "MonitorConfig":
        i = 1
        print_to_user(f"开始配置任务<{self.task_name}>")
        print_to_user(
            "聊天chat id和用户user id均同时支持整数id和字符串username, username必须以@开头，如@neo"
        )
        match_cfgs = []
        while True:
            print_to_user(f"\n配置第{i}个监控项")
            try:
                match_cfgs.append(self.ask_one())
            except Exception as e:
                print_to_user(e)
                print_to_user("配置失败")
                i -= 1
            continue_ = input("继续配置？(y/N)：")
            if continue_.strip().lower() != "y":
                break
            i += 1
        config = MonitorConfig(match_cfgs=match_cfgs)
        if config.requires_ai:
            print_to_user(OPENAI_USE_PROMPT)
        return config

    @classmethod
    async def udp_forward(cls, f: UDPForward, message: Message):
        data = str(message).encode("utf-8")
        loop = asyncio.get_running_loop()
        transport, protocol = await loop.create_datagram_endpoint(
            lambda: _UDPProtocol(), remote_addr=(f.host, f.port)
        )
        try:
            transport.sendto(data)
        finally:
            transport.close()

    @classmethod
    async def http_api_callback(cls, f: HttpCallback, message: Message):
        headers = f.headers or {}
        headers.update({"Content-Type": "application/json"})
        content = str(message).encode("utf-8")
        async with httpx.AsyncClient() as client:
            await client.post(
                str(f.url),
                content=content,
                headers=headers,
                timeout=10,
            )

    async def forward_to_external(self, match_cfg: MatchConfig, message: Message):
        if not match_cfg.external_forwards:
            return
        for forward in match_cfg.external_forwards:
            self.log(f"转发消息至{forward}")
            if isinstance(forward, UDPForward):
                asyncio.create_task(
                    self.udp_forward(
                        forward,
                        message,
                    )
                )
            elif isinstance(forward, HttpCallback):
                asyncio.create_task(
                    self.http_api_callback(
                        forward,
                        message,
                    )
                )

    async def on_message(self, client, message: Message):
        for match_cfg in self.config.match_cfgs:
            if not match_cfg.match(message):
                continue
            self.log(f"匹配到监控项：{match_cfg}")
            await self.forward_to_external(match_cfg, message)
            try:
                send_text = await self.get_send_text(match_cfg, message)
                if not send_text:
                    self.log("发送内容为空", level="WARNING")
                else:
                    forward_to_chat_id = match_cfg.forward_to_chat_id or message.chat.id
                    self.log(f"发送文本：{send_text}至{forward_to_chat_id}")
                    await self.send_message(
                        forward_to_chat_id,
                        send_text,
                        delete_after=match_cfg.delete_after,
                    )

                if match_cfg.push_via_server_chan:
                    server_chan_send_key = (
                        match_cfg.server_chan_send_key
                        or os.environ.get("SERVER_CHAN_SEND_KEY")
                    )
                    if not server_chan_send_key:
                        self.log("未配置Server酱的SendKey", level="WARNING")
                    else:
                        await sc_send(
                            server_chan_send_key,
                            f"匹配到监控项：{match_cfg.chat_id}",
                            f"消息内容为:\n\n{message.text}",
                        )
            except IndexError as e:
                logger.exception(e)

    async def get_send_text(self, match_cfg: MatchConfig, message: Message) -> str:
        send_text = match_cfg.get_send_text(message.text)
        if match_cfg.ai_reply and match_cfg.ai_prompt:
            send_text = await self.get_ai_tools().get_reply(
                match_cfg.ai_prompt,
                message.text,
            )
        return send_text

    async def run(self, num_of_dialogs=20):
        if self.user is None:
            await self.login(num_of_dialogs, print_chat=True)

        cfg = self.load_config(self.cfg_cls)
        if cfg.requires_ai:
            self.ensure_ai_cfg()

        self.app.add_handler(
            MessageHandler(self.on_message, filters.text & filters.chat(cfg.chat_ids)),
        )
        async with self.app:
            self.log("开始监控...")
            await idle()


class _UDPProtocol(asyncio.DatagramProtocol):
    """内部使用的UDP协议处理类"""

    def __init__(self):
        self.transport = None

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, addr):
        pass  # 不需要处理接收的数据

    def error_received(self, exc):
        print(f"UDP error received: {exc}")
