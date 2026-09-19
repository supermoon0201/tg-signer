import inspect

from pyrogram import raw

try:
    from pyrogram.types import AnimatedChatPhoto
except ImportError:  # Kurigram versions before AnimatedChatPhoto was introduced.
    AnimatedChatPhoto = None


_animated_chat_photo_parser_patched = False


def patch_animated_chat_photo_parser() -> None:
    """Ignore animated photos without a usable video representation.

    Kurigram 2.2.25 checks that ``photo.video_sizes`` is non-empty, but then
    calls ``max()`` after filtering the list to ``VideoSize`` instances. A
    photo containing only markup entries therefore raises ``ValueError``.
    """
    global _animated_chat_photo_parser_patched

    if AnimatedChatPhoto is None or _animated_chat_photo_parser_patched:
        return

    original_parse = AnimatedChatPhoto._parse

    async def safe_parse(client, photo):
        if isinstance(photo, raw.types.Photo) and not any(
            isinstance(video_size, raw.types.VideoSize)
            for video_size in photo.video_sizes or []
        ):
            return None

        parsed_photo = original_parse(client, photo)
        if inspect.isawaitable(parsed_photo):
            parsed_photo = await parsed_photo
        return parsed_photo

    AnimatedChatPhoto._parse = staticmethod(safe_parse)
    _animated_chat_photo_parser_patched = True
