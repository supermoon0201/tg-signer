from .methods import SafeGetForumTopics
from .patches import patch_animated_chat_photo_parser

patch_animated_chat_photo_parser()

__all__ = ["SafeGetForumTopics", "patch_animated_chat_photo_parser"]
