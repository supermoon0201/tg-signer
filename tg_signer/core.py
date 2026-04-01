import asyncio
import base64
import json
import logging
import os
import pathlib
import random
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from datetime import time as dt_time
from typing import (
    Annotated,
    Any,
    Awaitable,
    BinaryIO,
    Callable,
    Generic,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
)
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
from pyrogram.storage import SQLiteStorage
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
    OpenWebAppByTextAction,
    ReplyByCalculationProblemAction,
    SendDiceAction,
    SendTextAction,
    SignChatV3,
    SignConfigV3,
    SupportAction,
    UDPForward,
    WebViewCheckinAction,
)

from ._kurigram import SafeGetForumTopics
from .ai_tools import AITools, OpenAIConfigManager
from .notification.bark import bark_send
from .notification.server_chan import sc_send
from .sign_record_store import SignRecordStore
from .utils import UserInput, print_to_user

logger = logging.getLogger("tg-signer")

DICE_EMOJIS = ("🎲", "🎯", "🏀", "⚽", "🎳", "🎰")
TURNSTILE_HOOK_SCRIPT = """
(() => {
  if (window.__tgSignerTurnstileHookInstalled) {
    return;
  }
  window.__tgSignerTurnstileHookInstalled = true;
  window.__tgSignerTurnstile = {
    renders: [],
    lastToken: null,
    callback: null,
  };

  const wrapCallback = (cb) => {
    if (typeof cb !== "function") {
      return cb;
    }
    return function(token) {
      window.__tgSignerTurnstile.lastToken = token || null;
      window.__tgSignerTurnstile.callback = cb;
      return cb(token);
    };
  };

  const install = () => {
    if (!window.turnstile || window.turnstile.__tgSignerWrapped) {
      return;
    }

    const originalRender = window.turnstile.render.bind(window.turnstile);
    window.turnstile.render = function(container, params) {
      try {
        const recorded = {
          sitekey: params?.sitekey || null,
          action: params?.action || null,
          data: params?.cData || null,
          pagedata: params?.chlPageData || null,
        };
        window.__tgSignerTurnstile.renders.push(recorded);
        if (typeof params?.callback === "function") {
          window.__tgSignerTurnstile.callback = params.callback;
          params = { ...params, callback: wrapCallback(params.callback) };
        }
      } catch (e) {
        console.debug("tg-signer turnstile hook failed", e);
      }
      return originalRender(container, params);
    };

    window.turnstile.__tgSignerWrapped = true;
  };

  install();
  const timer = window.setInterval(install, 50);
  window.setTimeout(() => window.clearInterval(timer), 10000);
})();
"""

Session.START_TIMEOUT = 5  # 原始超时时间为2秒，但一些代理访问会超时，所以这里调大一点

OPENAI_USE_PROMPT = "当前任务需要配置大模型，请确保运行前正确设置`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`等环境变量，或通过`tg-signer llm-config`持久化配置。"

CHAT_TYPE_LABELS = {
    ChatType.BOT: "BOT",
    ChatType.GROUP: "群组",
    ChatType.SUPERGROUP: "超级群组",
    ChatType.CHANNEL: "频道",
    ChatType.FORUM: "论坛群组",
    ChatType.DIRECT: "频道私信",
}


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
    type_ = CHAT_TYPE_LABELS.get(chat.type, "个人")

    none_or_dash = lambda x: x or "-"  # noqa: E731

    return f"id: {chat.id}, username: {none_or_dash(chat.username)}, title: {none_or_dash(chat.title)}, type: {type_}, name: {none_or_dash(chat.first_name)}"


def chat_has_forum_topics(chat: Chat) -> bool:
    return chat.type == ChatType.FORUM or (
        chat.type == ChatType.SUPERGROUP and chat.is_forum
    )


def readable_topic(topic) -> str:
    none_or_dash = lambda x: x or "-"  # noqa: E731
    return (
        f"message_thread_id: {topic.id}, title: {none_or_dash(topic.title)}, "
        f"closed: {bool(getattr(topic, 'is_closed', False))}, "
        f"pinned: {bool(getattr(topic, 'is_pinned', False))}"
    )


_CLIENT_INSTANCES: dict[str, "Client"] = {}

# reference counts and async locks for shared client lifecycle management
# Keyed by account name. Use asyncio locks to serialize start/stop operations
# so multiple coroutines in the same process can safely share one Client.
_CLIENT_REFS: defaultdict[str, int] = defaultdict(int)
_CLIENT_ASYNC_LOCKS: dict[str, asyncio.Lock] = {}

# login bootstrap state keyed by account key. This prevents concurrent tasks
# from repeatedly calling get_me/get_dialogs for the same account.
_LOGIN_ASYNC_LOCKS: dict[str, asyncio.Lock] = {}
_LOGIN_USERS: dict[str, User] = {}

_API_ASYNC_LOCKS: dict[str, asyncio.Lock] = {}
_API_LAST_CALL_AT: dict[str, float] = {}
_API_MIN_INTERVAL_SECONDS = 0.35
_API_FLOODWAIT_PADDING_SECONDS = 0.5
_API_MAX_FLOODWAIT_RETRIES = 2

RouteKey = tuple[int, Optional[int]]


class Client(SafeGetForumTopics, BaseClient):
    def __init__(self, name: str, *args, **kwargs):
        key = kwargs.pop("key", None)
        super().__init__(name, *args, **kwargs)
        self.key = key or str(pathlib.Path(self.workdir).joinpath(self.name).resolve())
        if self.in_memory and not self.session_string:
            self.load_session_string()
            self.storage = SQLiteStorage(
                name=self.name,
                workdir=self.workdir,
                session_string=self.session_string,
                in_memory=True,
            )

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

    async def log_out(self):
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
ApiCallResultT = TypeVar("ApiCallResultT")


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

    async def _call_telegram_api(
        self,
        operation: str,
        call: Callable[[], Awaitable[ApiCallResultT]],
        *,
        retry_on_floodwait: bool = True,
    ) -> ApiCallResultT:
        key = self.app.key
        lock = _API_ASYNC_LOCKS.get(key)
        if lock is None:
            lock = asyncio.Lock()
            _API_ASYNC_LOCKS[key] = lock

        retries_left = _API_MAX_FLOODWAIT_RETRIES
        while True:
            async with lock:
                loop = asyncio.get_running_loop()
                last_called_at = _API_LAST_CALL_AT.get(key)
                if last_called_at is not None:
                    wait_for = _API_MIN_INTERVAL_SECONDS - (
                        loop.time() - last_called_at
                    )
                    if wait_for > 0:
                        await asyncio.sleep(wait_for)
                try:
                    result = await call()
                    _API_LAST_CALL_AT[key] = loop.time()
                    return result
                except errors.FloodWait as e:
                    _API_LAST_CALL_AT[key] = loop.time()
                    if not retry_on_floodwait or retries_left <= 0:
                        raise
                    retries_left -= 1
                    wait_seconds = (
                        max(float(getattr(e, "value", 0) or 0), 0)
                        + _API_FLOODWAIT_PADDING_SECONDS
                    )
                    self.log(
                        f"{operation} 触发 FloodWait，等待 {wait_seconds:.1f}s 后重试（剩余重试 {retries_left} 次）",
                        level="WARNING",
                    )
                    await asyncio.sleep(wait_seconds)

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
        key = app.key
        lock = _LOGIN_ASYNC_LOCKS.get(key)
        if lock is None:
            lock = asyncio.Lock()
            _LOGIN_ASYNC_LOCKS[key] = lock

        async with lock:
            me = _LOGIN_USERS.get(key)
            if me is None:
                async with app:
                    me = await self._call_telegram_api("users.GetFullUser", app.get_me)

                    async def load_latest_chats():
                        chats = []
                        latest_chats = []
                        async for dialog in app.get_dialogs(limit=num_of_dialogs):
                            chat = dialog.chat
                            chats.append(chat)
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
                        return chats, latest_chats

                    chats, latest_chats = await self._call_telegram_api(
                        "messages.GetDialogs", load_latest_chats
                    )

                    if print_chat:
                        for chat in chats:
                            print_to_user(readable_chat(chat))
                            if chat_has_forum_topics(chat):
                                try:
                                    topics = await asyncio.wait_for(
                                        self.get_forum_topics(chat.id, limit=20),
                                        timeout=5,
                                    )
                                    for topic in topics:
                                        print_to_user(f"  {readable_topic(topic)}")
                                except (asyncio.TimeoutError, errors.RPCError):
                                    # Keep login robust: many chats don't support
                                    # forum topics or the current account may not
                                    # have permissions to read them.
                                    pass

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
                    await self._call_telegram_api(
                        "auth.ExportAuthorization", self.app.save_session_string
                    )
                _LOGIN_USERS[key] = me
            else:
                self.log("检测到同账号已完成登录初始化，复用已有会话信息")
            self.set_me(me)

    async def logout(self):
        self.log("开始登出...")
        is_authorized = await self.app.connect()
        if not is_authorized:
            await self.app.storage.delete()
            _LOGIN_USERS.pop(self.app.key, None)
            self.user = None
            return None
        result = await self.app.log_out()
        _LOGIN_USERS.pop(self.app.key, None)
        self.user = None
        return result

    async def send_message(
        self,
        chat_id: Union[int, str],
        text: str,
        delete_after: int = None,
        message_thread_id: Optional[int] = None,
        **kwargs,
    ):
        """
        发送文本消息
        :param chat_id:
        :param text:
        :param delete_after: 秒, 发送消息后进行删除，``None`` 表示不删除, ``0`` 表示立即删除.
        :param kwargs:
        :return:
        """
        send_kwargs = dict(kwargs)
        if message_thread_id is not None:
            send_kwargs["message_thread_id"] = message_thread_id
        message = await self._call_telegram_api(
            "messages.SendMessage",
            lambda: self.app.send_message(chat_id, text, **send_kwargs),
        )
        if delete_after is not None:
            self.log(
                f"Message「{text}」 to {chat_id} will be deleted after {delete_after} seconds."
            )
            self.log("Waiting...")
            await asyncio.sleep(delete_after)
            await self._call_telegram_api("messages.DeleteMessages", message.delete)
            self.log(f"Message「{text}」 to {chat_id} deleted!")
        return message

    async def send_dice(
        self,
        chat_id: Union[int, str],
        emoji: str = "🎲",
        delete_after: int = None,
        message_thread_id: Optional[int] = None,
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
        send_kwargs = dict(kwargs)
        if message_thread_id is not None:
            send_kwargs["message_thread_id"] = message_thread_id
        message = await self._call_telegram_api(
            "messages.SendMedia",
            lambda: self.app.send_dice(chat_id, emoji, **send_kwargs),
        )
        if message and delete_after is not None:
            self.log(
                f"Dice「{emoji}」 to {chat_id} will be deleted after {delete_after} seconds."
            )
            self.log("Waiting...")
            await asyncio.sleep(delete_after)
            await self._call_telegram_api("messages.DeleteMessages", message.delete)
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

    async def get_forum_topics(self, chat_id: Union[int, str], limit: int = 20):
        topics = []

        async def _collect_topics():
            async for topic in self.app.get_forum_topics(chat_id, limit=limit):
                topics.append(topic)
            return topics

        return await self._call_telegram_api("channels.GetForumTopics", _collect_topics)

    async def list_topics(self, chat_id: Union[int, str], limit: int = 20):
        if self.user is None:
            await self.login(print_chat=False)
        async with self.app:
            try:
                topics = await self.get_forum_topics(chat_id, limit=limit)
            except errors.RPCError as e:
                print_to_user(f"获取话题失败: {e}")
                return []
            if not topics:
                print_to_user("未获取到话题，可能该聊天未开启话题或无权限。")
                return []
            for topic in topics:
                print_to_user(readable_topic(topic))
            return topics

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
    sign_chats: defaultdict[RouteKey, list[SignChatV3]]  # 签到配置列表
    chat_messages: defaultdict[
        RouteKey,
        Annotated[
            dict[int, Optional[Message]],
            Field(default_factory=dict),
        ],
    ]  # 收到的消息，key为(chat id, message_thread_id)
    waiting_message: Optional[Message]  # 正在处理的消息


class UserSigner(BaseUserWorker[SignConfigV3]):
    _workdir = ".signer"
    _tasks_dir = "signs"
    cfg_cls = SignConfigV3
    context: UserSignerWorkerContext
    _DEFAULT_REPEAT_SUCCESS_KEYWORDS = ("签到成功", "签到日期", "当前持有")
    _DEFAULT_REPEAT_DONE_KEYWORDS = (
        "已经签过到了",
        "今天已经签过到了",
        "已经签到",
        "今日已签到",
        "今朝已至",
        "签到是无聊的活动哦",
    )

    def ensure_ctx(self) -> UserSignerWorkerContext:
        return UserSignerWorkerContext(
            waiter=Waiter(),
            sign_chats=defaultdict(list),
            chat_messages=defaultdict(dict),
            waiting_message=None,
        )

    @property
    def sign_record_store(self) -> SignRecordStore:
        return SignRecordStore(self.workdir)

    @staticmethod
    def get_route_key(
        chat_id: int, message_thread_id: Optional[int] = None
    ) -> RouteKey:
        return chat_id, message_thread_id

    @property
    def sign_record_file(self):
        sign_record_dir = self.task_dir / str(self.user.id)
        make_dirs(sign_record_dir)
        return sign_record_dir / "sign_record.json"

    @property
    def legacy_sign_record_file(self):
        return self.task_dir / "sign_record.json"

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
                elif action == SupportAction.OPEN_WEBAPP_BY_TEXT:
                    text_of_btn_to_click = local_input_(
                        "Telegram消息中要点击的小程序按钮文本: "
                    )
                    page_button_text = local_input_(
                        "小程序页面中要点击的按钮文本: "
                    )
                    ready_text = local_input_(
                        "点击前需要等待出现的文本（如 验证成功，可选，直接回车跳过）: "
                    ).strip()
                    response_url_contains = local_input_(
                        "点击后需要监听的接口URL片段（如 /auth/checkin/submit，可选）: "
                    ).strip()
                    success_text = local_input_(
                        "点击后期望出现的成功文本（可选，直接回车跳过）: "
                    ).strip()
                    turnstile_enabled = (
                        local_input_(
                            "是否处理 Cloudflare Turnstile 验证？(y/N): "
                        ).strip().lower()
                        == "y"
                    )
                    turnstile_use_2captcha = False
                    if turnstile_enabled:
                        turnstile_use_2captcha = (
                            local_input_(
                                "Turnstile 自动点击失败后是否使用 2captcha？(y/N): "
                            ).strip().lower()
                            == "y"
                        )
                    captcha_image_selector = local_input_(
                        "验证码图片选择器（可选，留空则不启用TwoCaptcha）: "
                    ).strip()
                    captcha_input_selector = None
                    captcha_submit_selector = None
                    captcha_success_text = None
                    two_captcha_api_key = None
                    if captcha_image_selector:
                        captcha_input_selector = local_input_(
                            "验证码输入框选择器: "
                        ).strip()
                        captcha_submit_selector = (
                            local_input_(
                                "验证码提交按钮选择器（可选，直接回车跳过）: "
                            ).strip()
                            or None
                        )
                        captcha_success_text = (
                            local_input_(
                                "验证码成功文本（可选，直接回车跳过）: "
                            ).strip()
                            or None
                        )
                        two_captcha_api_key = (
                            local_input_(
                                "2captcha API Key（可选，直接回车则改用环境变量TWOCAPTCHA_API_KEY）: "
                            ).strip()
                            or None
                        )
                    actions.append(
                        OpenWebAppByTextAction(
                            text=text_of_btn_to_click,
                            page_button_text=page_button_text,
                            ready_text=ready_text or None,
                            response_url_contains=response_url_contains or None,
                            success_text=success_text or None,
                            turnstile_enabled=turnstile_enabled,
                            turnstile_use_2captcha=turnstile_use_2captcha,
                            captcha_image_selector=captcha_image_selector or None,
                            captcha_input_selector=captcha_input_selector or None,
                            captcha_submit_selector=captcha_submit_selector,
                            captcha_success_text=captcha_success_text,
                            two_captcha_api_key=two_captcha_api_key,
                        )
                    )
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
        use_message_thread = (
            input_("是否发送到话题（message_thread_id）？(y/N)：").strip().lower()
            == "y"
        )
        message_thread_id = None
        if use_message_thread:
            message_thread_id = int(input_("message_thread_id: "))
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
            "message_thread_id": message_thread_id,
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

    def _validate_sign_at(
        self,
        sign_at_str: str,
    ) -> Optional[str]:
        sign_at_str = sign_at_str.replace("：", ":").strip()

        try:
            sign_at = dt_time.fromisoformat(sign_at_str)
            crontab_expr = self._time_to_crontab(sign_at)
        except ValueError:
            try:
                croniter(sign_at_str)
                crontab_expr = sign_at_str
            except CroniterBadCronError:
                self.log(f"时间格式错误: {sign_at_str}", level="error")
                return None
        return crontab_expr

    @staticmethod
    def _time_to_crontab(sign_at: dt_time) -> str:
        return f"{sign_at.minute} {sign_at.hour} * * *"

    def load_sign_record(self):
        user_id = str(self.user.id)
        store = self.sign_record_store
        if not store.has_records(self.task_name, user_id):
            # Import legacy JSON lazily so existing workdirs keep working
            # without requiring an explicit migration step first.
            imported_paths = []
            if store.import_json_file(
                self.task_name,
                user_id,
                self.sign_record_file,
                account=self._account,
            ):
                imported_paths.append(self.sign_record_file)
            if store.import_json_file(
                self.task_name,
                user_id,
                self.legacy_sign_record_file,
                account=self._account,
            ):
                imported_paths.append(self.legacy_sign_record_file)
            if imported_paths:
                joined_paths = ", ".join(str(path) for path in imported_paths)
                self.log(
                    f"检测到旧版签到记录文件，已自动导入 SQLite: {joined_paths}。建议执行 `tg-signer migrate-sign-records` 统一迁移历史记录。",
                    level="WARNING",
                )
        return store.load_records(self.task_name, user_id)

    def persist_sign_record(
        self, sign_record: dict[str, str], sign_date: str, signed_at: str
    ) -> None:
        sign_record[sign_date] = signed_at
        self.sign_record_store.upsert_record(
            self.task_name,
            str(self.user.id),
            sign_date,
            signed_at,
            account=self._account,
        )

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
                route_key = self.get_route_key(chat.chat_id, chat.message_thread_id)
                self.context.sign_chats[route_key].append(chat)
                try:
                    await self.sign_a_chat(chat)
                except errors.RPCError as _e:
                    self.log(f"签到失败: {_e} \nchat: \n{chat}")
                    logger.warning(_e, exc_info=True)
                    continue

                self.context.chat_messages[route_key].clear()
                await asyncio.sleep(config.sign_interval)
            self.persist_sign_record(sign_record, str(now.date()), now.isoformat())

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
        self,
        chat_id: int,
        text: str,
        delete_after: int = None,
        message_thread_id: Optional[int] = None,
        **kwargs,
    ):
        if self.user is None:
            await self.login(print_chat=False)
        async with self.app:
            await self.send_message(
                chat_id,
                text,
                delete_after,
                message_thread_id=message_thread_id,
                **kwargs,
            )

    async def send_dice_cli(
        self,
        chat_id: Union[str, int],
        emoji: str = "🎲",
        delete_after: int = None,
        message_thread_id: Optional[int] = None,
        **kwargs,
    ):
        if self.user is None:
            await self.login(print_chat=False)
        async with self.app:
            await self.send_dice(
                chat_id,
                emoji,
                delete_after,
                message_thread_id=message_thread_id,
                **kwargs,
            )

    async def _on_message(self, client: Client, message: Message):
        message_thread_id = getattr(message, "message_thread_id", None)
        route_key = self.get_route_key(message.chat.id, message_thread_id)
        chats = self.context.sign_chats.get(route_key)
        if not chats and message_thread_id is not None:
            route_key = self.get_route_key(message.chat.id, None)
            chats = self.context.sign_chats.get(route_key)
        if not chats:
            self.log("忽略意料之外的聊天", level="WARNING")
            return
        self.context.chat_messages[route_key][message.id] = message

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
        found = self._find_callback_button(message, action.text)
        if not found:
            return False

        if not action.repeat_until_complete:
            btn = found[1]
            self.log(f"点击按钮: {btn.text}")
            answer = await self.request_callback_answer(
                self.app,
                message.chat.id,
                message.id,
                btn.callback_data,
            )
            return answer is not None

        route_key = self.get_route_key(
            message.chat.id,
            getattr(message, "message_thread_id", None),
        )
        clicked = False
        start = time.perf_counter()

        while time.perf_counter() - start < action.repeat_timeout:
            latest_found = self._find_latest_callback_button(route_key, action.text)
            if latest_found is None:
                if clicked:
                    self.log(f"按钮「{action.text}」已消失，结束连续点击")
                return clicked

            target_message, btn = latest_found
            self.log(f"连续点击按钮: {btn.text}")
            answer = await self.request_callback_answer(
                self.app,
                target_message.chat.id,
                target_message.id,
                btn.callback_data,
            )
            if answer and self._callback_answer_matches_terminal_state(action, answer):
                return True
            if answer is None:
                return clicked

            clicked = True
            terminal = await self._wait_for_click_completion_state(
                route_key, action, action.repeat_interval
            )
            if terminal is not None:
                return terminal

        self.log(
            f"按钮「{action.text}」连续点击达到超时上限 {action.repeat_timeout} 秒",
            level="WARNING",
        )
        return clicked

    async def _wait_for_click_completion_state(
        self, route_key: RouteKey, action: ClickKeyboardByTextAction, timeout: float
    ) -> Optional[bool]:
        start = time.perf_counter()
        while time.perf_counter() - start < timeout:
            state = self._get_click_completion_state(route_key, action)
            if state is not None:
                return state
            await asyncio.sleep(0.2)
        return None

    def _get_click_completion_state(
        self, route_key: RouteKey, action: ClickKeyboardByTextAction
    ) -> Optional[bool]:
        messages_dict = self.context.chat_messages.get(route_key, {})
        for message in reversed(list(messages_dict.values())):
            if not message:
                continue
            text = self._normalize_match_text(self._message_match_text(message))
            if self._matches_done_text(action, text):
                self.log(f"检测到已签到提示，结束连续点击: {text}")
                return True
            if self._matches_success_text(action, text):
                self.log(f"检测到签到成功提示，结束连续点击: {text}")
                return True

        latest_found = self._find_latest_callback_button(route_key, action.text)
        if latest_found is None:
            return True
        return None

    def _find_callback_button(
        self, message: Optional[Message], text: str
    ) -> Optional[tuple[Message, InlineKeyboardButton]]:
        if not message:
            return None
        reply_markup = getattr(message, "reply_markup", None)
        if not isinstance(reply_markup, InlineKeyboardMarkup):
            return None
        for btn in (b for row in reply_markup.inline_keyboard for b in row if b.text):
            if text in btn.text and btn.callback_data:
                return message, btn
        return None

    def _find_latest_callback_button(
        self, route_key: RouteKey, text: str
    ) -> Optional[tuple[Message, InlineKeyboardButton]]:
        messages_dict = self.context.chat_messages.get(route_key, {})
        for message in reversed(list(messages_dict.values())):
            found = self._find_callback_button(message, text)
            if found:
                return found
        return None

    def _message_match_text(self, message: Message) -> str:
        parts = []
        for attr in ("text", "caption"):
            value = getattr(message, attr, None)
            if value:
                parts.append(str(value))

        reply_markup = getattr(message, "reply_markup", None)
        if isinstance(reply_markup, InlineKeyboardMarkup):
            for btn in (b for row in reply_markup.inline_keyboard for b in row if b.text):
                parts.append(btn.text)

        return "\n".join(parts)

    def _normalize_match_text(self, text: Optional[str]) -> str:
        if not text:
            return ""
        for ch in ("\u200b", "\u200c", "\u200d", "\u2060", "\ufeff", " "):
            text = text.replace(ch, "")
        return "".join(text.split()).lower()

    def _matches_success_text(
        self, action: ClickKeyboardByTextAction, normalized_text: str
    ) -> bool:
        keywords = [action.success_text] if action.success_text else []
        if not keywords:
            keywords = list(self._DEFAULT_REPEAT_SUCCESS_KEYWORDS)
        return any(
            self._normalize_match_text(keyword) in normalized_text
            for keyword in keywords
            if keyword
        )

    def _matches_done_text(
        self, action: ClickKeyboardByTextAction, normalized_text: str
    ) -> bool:
        keywords = [action.already_done_text] if action.already_done_text else []
        if not keywords:
            keywords = list(self._DEFAULT_REPEAT_DONE_KEYWORDS)
        return any(
            self._normalize_match_text(keyword) in normalized_text
            for keyword in keywords
            if keyword
        )

    def _callback_answer_matches_terminal_state(
        self, action: ClickKeyboardByTextAction, answer
    ) -> bool:
        normalized_text = self._normalize_match_text(getattr(answer, "message", None))
        if not normalized_text:
            return False
        if self._matches_done_text(action, normalized_text):
            self.log(f"检测到已签到弹窗，结束连续点击: {getattr(answer, 'message', '')}")
            return True
        if self._matches_success_text(action, normalized_text):
            self.log(f"检测到签到成功弹窗，结束连续点击: {getattr(answer, 'message', '')}")
            return True
        return False

    async def _get_webview_url_from_button(
        self, message: Message, button: InlineKeyboardButton
    ) -> Optional[str]:
        from pyrogram.raw.functions.messages import RequestWebView

        raw_url = None
        if button.web_app:
            raw_url = button.web_app.url
        elif button.url:
            raw_url = button.url
        elif button.login_url:
            raw_url = button.login_url.url

        if not raw_url:
            return None

        bot_id = None
        if message.from_user and message.from_user.is_bot:
            bot_id = message.from_user.id
        elif getattr(message.chat, "id", None):
            bot_id = message.chat.id

        if bot_id is None:
            self.log("无法识别 WebApp 对应的 Bot", level="WARNING")
            return raw_url

        try:
            chat_peer = await self.app.resolve_peer(message.chat.id)
            bot_peer = await self.app.resolve_peer(bot_id)
            return (
                await self.app.invoke(
                    RequestWebView(
                        peer=chat_peer,
                        bot=bot_peer,
                        platform="ios",
                        url=raw_url,
                    )
                )
            ).url
        except Exception as e:
            self.log(
                f"获取 WebApp 认证链接失败，将回退到原始链接: {e}",
                level="WARNING",
            )
            return raw_url

    async def _run_webapp_page_action(
        self, action: OpenWebAppByTextAction, webview_url: str
    ) -> bool:
        try:
            from playwright.async_api import TimeoutError as PlaywrightTimeoutError
            from playwright.async_api import async_playwright
        except ImportError:
            self.log(
                "缺少 playwright 依赖，无法执行小程序页面自动点击。"
                "请安装 `playwright` 并执行 `playwright install chromium`。",
                level="ERROR",
            )
            return False

        try:
            async with async_playwright() as playwright:
                browser = await playwright.chromium.launch(headless=action.headless)
                page = await browser.new_page()
                try:
                    if action.turnstile_enabled:
                        await page.add_init_script(TURNSTILE_HOOK_SCRIPT)

                    response_future = None
                    if action.response_url_contains:
                        response_future = asyncio.get_running_loop().create_future()

                        async def on_response(resp):
                            if action.response_url_contains not in resp.url:
                                return
                            if response_future.done():
                                return
                            payload: dict[str, Any] | None = None
                            text = None
                            try:
                                payload = await resp.json()
                            except Exception:
                                try:
                                    text = await resp.text()
                                except Exception:
                                    text = None
                            response_future.set_result(
                                {
                                    "status": resp.status,
                                    "url": resp.url,
                                    "json": payload,
                                    "text": text,
                                }
                            )

                        page.on("response", on_response)

                    self.log("正在打开 WebApp 页面...")
                    await page.goto(
                        webview_url,
                        wait_until="domcontentloaded",
                        timeout=action.page_load_timeout * 1000,
                    )

                    if action.ready_text:
                        self.log(f"等待 WebApp 就绪文本: {action.ready_text}")
                        await page.get_by_text(
                            action.ready_text, exact=False
                        ).first.wait_for(
                            state="visible",
                            timeout=action.ready_timeout * 1000,
                        )
                        self.log(f"检测到 WebApp 就绪文本: {action.ready_text}")

                    if not await self._maybe_solve_webapp_captcha(action, page):
                        return False

                    click_attempts = 2 if action.turnstile_retry_after_solve else 1
                    for _click_attempt in range(click_attempts):
                        button = page.get_by_role(
                            "button", name=action.page_button_text, exact=False
                        ).first
                        try:
                            await button.wait_for(
                                state="visible", timeout=action.button_timeout * 1000
                            )
                        except PlaywrightTimeoutError:
                            button = page.get_by_text(
                                action.page_button_text, exact=False
                            ).first
                            await button.wait_for(
                                state="visible", timeout=action.button_timeout * 1000
                            )

                        await button.click(timeout=action.button_timeout * 1000)
                        self.log(
                            f"已在 WebApp 中点击按钮: {action.page_button_text}"
                        )

                        turnstile_result = await self._handle_turnstile_after_button_click(
                            action, page
                        )
                        if turnstile_result == "retry":
                            self.log(
                                "Cloudflare Turnstile 已处理，准备重新点击业务按钮。"
                            )
                            continue
                        if turnstile_result == "blocked":
                            return False
                        break

                    if response_future is not None:
                        self.log(
                            f"等待 WebApp 接口响应: {action.response_url_contains}"
                        )
                        response_result = await asyncio.wait_for(
                            response_future,
                            timeout=action.response_timeout,
                        )
                        self.log(
                            f"检测到 WebApp 接口响应: {response_result['status']} "
                            f"{response_result['url']}"
                        )
                        payload = response_result.get("json")
                        if isinstance(payload, dict):
                            success_value = payload.get(action.response_success_key)
                            if success_value != action.response_success_value:
                                message = None
                                if action.response_message_key:
                                    message = payload.get(action.response_message_key)
                                self.log(
                                    "WebApp 接口返回失败: "
                                    f"{message or payload}",
                                    level="WARNING",
                                )
                                return False
                            self.log(
                                f"WebApp 接口返回成功: {payload.get(action.response_success_key)}"
                            )
                            return True
                        self.log(
                            "WebApp 接口响应不是JSON，无法判定成功: "
                            f"{response_result.get('text') or response_result}",
                            level="WARNING",
                        )
                        return False

                    if action.success_text:
                        await page.get_by_text(
                            action.success_text, exact=False
                        ).first.wait_for(
                            state="visible",
                            timeout=action.success_timeout * 1000,
                        )
                        self.log(
                            f"检测到 WebApp 成功提示文本: {action.success_text}"
                        )
                    return True
                finally:
                    await browser.close()
        except Exception as e:
            self.log(f"WebApp 页面操作失败: {e}", level="ERROR")
            return False

    def _get_twocaptcha_api_key(self, action: OpenWebAppByTextAction) -> Optional[str]:
        return (
            action.two_captcha_api_key
            or os.environ.get("TWOCAPTCHA_API_KEY")
            or os.environ.get("TWO_CAPTCHA_API_KEY")
        )

    async def _solve_twocaptcha_image(
        self,
        api_key: str,
        image_bytes: bytes,
        timeout_seconds: int,
        poll_interval_seconds: int,
    ) -> str:
        encoded_image = base64.b64encode(image_bytes).decode("ascii")
        async with httpx.AsyncClient(timeout=30.0) as client:
            submit_resp = await client.post(
                "https://2captcha.com/in.php",
                data={
                    "key": api_key,
                    "method": "base64",
                    "body": encoded_image,
                    "json": 1,
                },
            )
            submit_resp.raise_for_status()
            submit_payload = submit_resp.json()
            if submit_payload.get("status") != 1:
                raise ValueError(
                    f"2captcha 提交失败: {submit_payload.get('request') or submit_payload}"
                )

            captcha_id = str(submit_payload["request"])
            deadline = time.monotonic() + timeout_seconds
            while time.monotonic() < deadline:
                await asyncio.sleep(max(poll_interval_seconds, 1))
                result_resp = await client.get(
                    "https://2captcha.com/res.php",
                    params={
                        "key": api_key,
                        "action": "get",
                        "id": captcha_id,
                        "json": 1,
                    },
                )
                result_resp.raise_for_status()
                result_payload = result_resp.json()
                if result_payload.get("status") == 1:
                    result = str(result_payload["request"]).strip()
                    if not result:
                        raise ValueError("2captcha 返回了空验证码结果")
                    return result

                if result_payload.get("request") == "CAPCHA_NOT_READY":
                    continue

                raise ValueError(
                    f"2captcha 识别失败: {result_payload.get('request') or result_payload}"
                )

        raise TimeoutError(
            f"2captcha 在 {timeout_seconds} 秒内未返回识别结果"
        )

    async def _solve_twocaptcha_turnstile(
        self,
        api_key: str,
        website_url: str,
        website_key: str,
        action: Optional[str],
        c_data: Optional[str],
        chl_page_data: Optional[str],
        timeout_seconds: int,
        poll_interval_seconds: int,
    ) -> dict[str, Any]:
        task: dict[str, Any] = {
            "type": "TurnstileTaskProxyless",
            "websiteURL": website_url,
            "websiteKey": website_key,
        }
        if action:
            task["action"] = action
        if c_data:
            task["data"] = c_data
        if chl_page_data:
            task["pagedata"] = chl_page_data

        async with httpx.AsyncClient(timeout=30.0) as client:
            create_resp = await client.post(
                "https://api.2captcha.com/createTask",
                json={
                    "clientKey": api_key,
                    "task": task,
                },
            )
            create_resp.raise_for_status()
            create_payload = create_resp.json()
            if create_payload.get("errorId") != 0:
                raise ValueError(
                    "2captcha Turnstile 提交失败: "
                    f"{create_payload.get('errorDescription') or create_payload}"
                )

            task_id = create_payload["taskId"]
            deadline = time.monotonic() + timeout_seconds
            while time.monotonic() < deadline:
                await asyncio.sleep(max(poll_interval_seconds, 1))
                result_resp = await client.post(
                    "https://api.2captcha.com/getTaskResult",
                    json={
                        "clientKey": api_key,
                        "taskId": task_id,
                    },
                )
                result_resp.raise_for_status()
                result_payload = result_resp.json()
                if result_payload.get("errorId") != 0:
                    raise ValueError(
                        "2captcha Turnstile 识别失败: "
                        f"{result_payload.get('errorDescription') or result_payload}"
                    )
                if result_payload.get("status") == "processing":
                    continue
                if result_payload.get("status") == "ready":
                    solution = result_payload.get("solution") or {}
                    token = solution.get("token")
                    if not token:
                        raise ValueError(
                            "2captcha Turnstile 返回结果缺少 token"
                        )
                    return solution

        raise TimeoutError(
            f"2captcha 在 {timeout_seconds} 秒内未返回 Turnstile 结果"
        )

    async def _get_turnstile_params(self, page: Any) -> Optional[dict[str, Any]]:
        params = await page.evaluate(
            """() => {
                const state = window.__tgSignerTurnstile || {};
                const renders = Array.isArray(state.renders) ? state.renders : [];
                const last = renders.length ? renders[renders.length - 1] : null;
                const widget = document.querySelector('.cf-turnstile');
                const responseInput = document.querySelector(
                  'input[name="cf-turnstile-response"], textarea[name="cf-turnstile-response"]'
                );
                return {
                  sitekey: last?.sitekey || widget?.dataset?.sitekey || null,
                  action: last?.action || widget?.dataset?.action || null,
                  data: last?.data || widget?.dataset?.cData || null,
                  pagedata: last?.pagedata || widget?.dataset?.chlPageData || null,
                  hasResponseInput: Boolean(responseInput),
                  token: state.lastToken || responseInput?.value || null,
                };
            }"""
        )
        if not params:
            return None
        if not params.get("sitekey") and not params.get("hasResponseInput"):
            return None
        return params

    async def _has_turnstile_token(self, page: Any) -> bool:
        return bool(
            await page.evaluate(
                """() => {
                    const state = window.__tgSignerTurnstile || {};
                    const token = state.lastToken
                      || document.querySelector(
                        'input[name="cf-turnstile-response"], textarea[name="cf-turnstile-response"]'
                      )?.value;
                    return Boolean(token);
                }"""
            )
        )

    async def _is_turnstile_visible(self, page: Any) -> bool:
        try:
            return bool(
                await page.evaluate(
                    """() => {
                        const iframe = document.querySelector(
                          'iframe[src*="challenges.cloudflare.com"]'
                        );
                        if (iframe) {
                          return true;
                        }
                        const widget = document.querySelector('.cf-turnstile');
                        if (!widget) {
                          return false;
                        }
                        const style = window.getComputedStyle(widget);
                        return style.display !== 'none' && style.visibility !== 'hidden';
                    }"""
                )
            )
        except Exception:
            return False

    async def _click_turnstile_checkbox(self, page: Any, timeout_seconds: int) -> bool:
        try:
            widget = page.locator(".cf-turnstile").first
            await widget.wait_for(state="visible", timeout=timeout_seconds * 1000)
            box = await widget.bounding_box()
            if box:
                click_x = box["x"] + min(max(box["width"] * 0.12, 18), 35)
                click_y = box["y"] + box["height"] / 2
                await page.mouse.click(click_x, click_y)
                self.log("已尝试点击 Turnstile 容器中的复选框区域。")
                return True
        except Exception:
            pass

        iframe_locator = page.frame_locator(
            'iframe[src*="challenges.cloudflare.com"]'
        ).first
        for selector in (
            'input[type="checkbox"]',
            '[role="checkbox"]',
            'label.ctp-checkbox-label',
            '.ctp-checkbox-label',
        ):
            try:
                checkbox = iframe_locator.locator(selector).first
                await checkbox.wait_for(
                    state="visible", timeout=timeout_seconds * 1000
                )
                await checkbox.click(timeout=timeout_seconds * 1000)
                self.log(f"已尝试点击 Turnstile 复选框: {selector}")
                return True
            except Exception:
                continue

        try:
            iframe = page.locator('iframe[src*="challenges.cloudflare.com"]').first
            box = await iframe.bounding_box(timeout=timeout_seconds * 1000)
            if box:
                click_x = box["x"] + min(max(box["width"] * 0.12, 18), 35)
                click_y = box["y"] + box["height"] / 2
                await page.mouse.click(click_x, click_y)
                self.log("已尝试按 Turnstile iframe 坐标点击复选框区域。")
                return True
        except Exception:
            pass

        return False

    async def _apply_turnstile_token(self, page: Any, token: str) -> bool:
        return bool(
            await page.evaluate(
                """(turnstileToken) => {
                    const state = window.__tgSignerTurnstile || {};
                    state.lastToken = turnstileToken;
                    let applied = false;
                    for (const el of document.querySelectorAll(
                      'input[name="cf-turnstile-response"], textarea[name="cf-turnstile-response"], '
                      + 'input[name="g-recaptcha-response"], textarea[name="g-recaptcha-response"]'
                    )) {
                      el.value = turnstileToken;
                      el.textContent = turnstileToken;
                      el.dispatchEvent(new Event('input', { bubbles: true }));
                      el.dispatchEvent(new Event('change', { bubbles: true }));
                      applied = true;
                    }
                    if (typeof state.callback === 'function') {
                      state.callback(turnstileToken);
                      applied = true;
                    }
                    return applied;
                }""",
                token,
            )
        )

    async def _wait_for_turnstile_passed(
        self, page: Any, timeout_seconds: int
    ) -> bool:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if await self._has_turnstile_token(page):
                return True
            if not await self._is_turnstile_visible(page):
                return True
            await asyncio.sleep(0.5)
        return False

    async def _solve_turnstile_with_2captcha(
        self, action: OpenWebAppByTextAction, page: Any
    ) -> bool:
        api_key = self._get_twocaptcha_api_key(action)
        if not api_key:
            self.log(
                "未配置 2captcha API Key，无法处理 Turnstile。",
                level="ERROR",
            )
            return False

        params = await self._get_turnstile_params(page)
        if not params or not params.get("sitekey"):
            self.log(
                "未能提取 Turnstile 的 sitekey，无法调用 2captcha。",
                level="WARNING",
            )
            return False

        self.log(
            "检测到 Cloudflare Turnstile，开始调用 2captcha。"
        )
        solution = await self._solve_twocaptcha_turnstile(
            api_key=api_key,
            website_url=page.url,
            website_key=params["sitekey"],
            action=params.get("action"),
            c_data=params.get("data"),
            chl_page_data=params.get("pagedata"),
            timeout_seconds=action.turnstile_timeout,
            poll_interval_seconds=action.captcha_poll_interval,
        )
        applied = await self._apply_turnstile_token(page, solution["token"])
        if not applied:
            self.log(
                "2captcha 已返回 Turnstile token，但页面未找到回填入口。",
                level="WARNING",
            )
            return False
        self.log("2captcha Turnstile token 已注入页面。")
        return await self._wait_for_turnstile_passed(page, action.turnstile_timeout)

    async def _handle_turnstile_after_button_click(
        self, action: OpenWebAppByTextAction, page: Any
    ) -> str:
        if not action.turnstile_enabled:
            return "done"

        await page.wait_for_timeout(1500)
        if not await self._is_turnstile_visible(page):
            return "done"

        self.log("检测到 Cloudflare Turnstile 验证。")

        if action.turnstile_auto_click:
            auto_clicked = await self._click_turnstile_checkbox(
                page, timeout_seconds=min(action.turnstile_timeout, 15)
            )
            if auto_clicked:
                if await self._wait_for_turnstile_passed(
                    page, action.turnstile_timeout
                ):
                    self.log("Cloudflare Turnstile 已自动通过。")
                    return "retry" if action.turnstile_retry_after_solve else "done"
                self.log(
                    "Turnstile 复选框已点击，但仍未通过验证。",
                    level="WARNING",
                )

        if action.turnstile_use_2captcha:
            solved = await self._solve_turnstile_with_2captcha(action, page)
            if solved:
                self.log("Cloudflare Turnstile 已通过 2captcha 处理。")
                return "retry" if action.turnstile_retry_after_solve else "done"

        self.log(
            "Cloudflare Turnstile 尚未通过，当前流程无法继续。",
            level="WARNING",
        )
        return "blocked"

    async def _maybe_solve_webapp_captcha(
        self, action: OpenWebAppByTextAction, page: Any
    ) -> bool:
        if not action.captcha_image_selector and not action.captcha_input_selector:
            return True

        if not action.captcha_image_selector or not action.captcha_input_selector:
            self.log(
                "已配置 WebApp 验证码识别，但缺少图片选择器或输入框选择器。",
                level="ERROR",
            )
            return False

        api_key = self._get_twocaptcha_api_key(action)
        if not api_key:
            self.log(
                "未配置 2captcha API Key。请在动作中填写，或设置环境变量"
                " `TWOCAPTCHA_API_KEY`。",
                level="ERROR",
            )
            return False

        image_locator = page.locator(action.captcha_image_selector).first
        input_locator = page.locator(action.captcha_input_selector).first

        await image_locator.wait_for(
            state="visible", timeout=action.button_timeout * 1000
        )
        await input_locator.wait_for(
            state="visible", timeout=action.button_timeout * 1000
        )
        self.log(
            f"检测到 WebApp 验证码，开始调用 2captcha: "
            f"{action.captcha_image_selector}"
        )
        image_bytes = await image_locator.screenshot(type="png")
        captcha_text = await self._solve_twocaptcha_image(
            api_key=api_key,
            image_bytes=image_bytes,
            timeout_seconds=action.captcha_timeout,
            poll_interval_seconds=action.captcha_poll_interval,
        )

        await input_locator.fill(captcha_text)
        self.log(f"2captcha 识别完成，已填入验证码: {captcha_text}")

        if action.captcha_submit_selector:
            submit_locator = page.locator(action.captcha_submit_selector).first
            await submit_locator.wait_for(
                state="visible", timeout=action.button_timeout * 1000
            )
            await submit_locator.click(timeout=action.button_timeout * 1000)
            self.log(
                f"已点击验证码提交按钮: {action.captcha_submit_selector}"
            )

        if action.captcha_success_text:
            await page.get_by_text(
                action.captcha_success_text, exact=False
            ).first.wait_for(
                state="visible",
                timeout=action.success_timeout * 1000,
            )
            self.log(f"检测到验证码成功文本: {action.captcha_success_text}")

        return True

    async def _open_webapp_by_text(
        self, action: OpenWebAppByTextAction, message: Message
    ) -> bool:
        if reply_markup := message.reply_markup:
            if isinstance(reply_markup, InlineKeyboardMarkup):
                for btn in (
                    b for row in reply_markup.inline_keyboard for b in row if b.text
                ):
                    if action.text not in btn.text:
                        continue
                    if not (btn.web_app or btn.url or btn.login_url):
                        self.log(
                            f"按钮「{btn.text}」不是 WebApp/URL 按钮，跳过",
                            level="WARNING",
                        )
                        return False
                    self.log(f"打开小程序按钮: {btn.text}")
                    webview_url = await self._get_webview_url_from_button(message, btn)
                    if not webview_url:
                        self.log("未能获取小程序链接", level="WARNING")
                        return False
                    return await self._run_webapp_page_action(action, webview_url)
        return False

    async def _reply_by_calculation_problem(
        self, action: ReplyByCalculationProblemAction, message
    ):
        if message.text:
            self.log("检测到文本回复，尝试调用大模型进行计算题回答")
            self.log(f"问题: \n{message.text}")
            answer = await self.get_ai_tools().calculate_problem(message.text)
            self.log(f"回答为: {answer}")
            await self.send_message(
                message.chat.id,
                answer,
                message_thread_id=getattr(message, "message_thread_id", None),
            )
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
                        "签到失败: 异常返回信息",
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
        route_key = self.get_route_key(chat.chat_id, chat.message_thread_id)
        if isinstance(action, SendTextAction):
            return await self.send_message(
                chat.chat_id,
                action.text,
                chat.delete_after,
                message_thread_id=chat.message_thread_id,
            )
        elif isinstance(action, SendDiceAction):
            return await self.send_dice(
                chat.chat_id,
                action.dice,
                chat.delete_after,
                message_thread_id=chat.message_thread_id,
            )
        elif isinstance(action, WebViewCheckinAction):
            return await self._webview_checkin(action)

        # 特殊处理GIF验证码场景：需要等待两条消息
        if isinstance(action, ChooseOptionByGifAction):
            return await self._wait_for_gif_action(chat, action, timeout)

        self.context.waiter.add(route_key)
        start = time.perf_counter()
        last_message = None
        while time.perf_counter() - start < timeout:
            await asyncio.sleep(0.3)
            messages_dict = self.context.chat_messages.get(route_key)
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
                elif isinstance(action, OpenWebAppByTextAction):
                    ok = await self._open_webapp_by_text(action, message)
                elif isinstance(action, ReplyByCalculationProblemAction):
                    ok = await self._reply_by_calculation_problem(action, message)
                elif isinstance(action, ChooseOptionByImageAction):
                    ok = await self._choose_option_by_image(action, message)
                if ok:
                    self.context.waiter.sub(route_key)
                    # 将消息ID对应value置为None，保证收到消息的编辑时消息所处的顺序
                    self.context.chat_messages[route_key][message.id] = None
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
        route_key = self.get_route_key(chat.chat_id, chat.message_thread_id)
        self.context.waiter.add(route_key)
        start = time.perf_counter()
        button_message = None
        gif_message = None
        processed_ids = set()

        while time.perf_counter() - start < timeout:
            await asyncio.sleep(0.3)
            messages_dict = self.context.chat_messages.get(route_key)
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
                        self.context.waiter.sub(route_key)
                        self.context.chat_messages[route_key][button_message.id] = (
                            None
                        )
                        self.context.chat_messages[route_key][gif_message.id] = None
                        return None
                    else:
                        # 识别失败，返回重试标记
                        self.context.waiter.sub(route_key)
                        self.context.chat_messages[route_key].clear()
                        return "RETRY_FLOW"

        self.log(
            f"等待GIF验证码超时: \nchat: \n{chat} \naction: {action}", level="WARNING"
        )
        self.context.waiter.sub(route_key)
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
            answer = await self._call_telegram_api(
                "messages.GetBotCallbackAnswer",
                lambda: client.request_callback_answer(
                    chat_id,
                    message_id,
                    callback_data=callback_data,
                    **kwargs,
                ),
            )
            msg = getattr(answer, "message", None)
            if msg:
                self.log(f"点击完成，弹窗消息: {msg}")
            else:
                self.log("点击完成")
            return answer
        except (errors.BadRequest, TimeoutError) as e:
            err_text = str(e)
            if "MESSAGE_ID_INVALID" in err_text:
                self.log("按钮对应消息已失效，停止继续点击")
            else:
                self.log(e, level="ERROR")
            return None

    async def schedule_messages(
        self,
        chat_id: Union[int, str],
        text: str,
        crontab: str = None,
        next_times: int = 1,
        random_seconds: int = 0,
        message_thread_id: Optional[int] = None,
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
                await self._call_telegram_api(
                    "messages.SendScheduledMessage",
                    lambda schedule_date=next_dt: self.app.send_message(
                        chat_id,
                        text,
                        schedule_date=schedule_date,
                        message_thread_id=message_thread_id,
                    ),
                )
                await asyncio.sleep(0.1)
                print_to_user(f"已配置次数：{n + 1}")
        self.log(f"已配置定时发送消息，次数{next_times}")
        return results

    async def get_schedule_messages(self, chat_id: Union[int, str]):
        if self.user is None:
            await self.login(print_chat=False)
        async with self.app:
            messages = await self._call_telegram_api(
                "messages.GetScheduledHistory",
                lambda: self.app.get_scheduled_messages(chat_id),
            )
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
