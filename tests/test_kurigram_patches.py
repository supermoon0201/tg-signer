import pytest
from pyrogram import raw

try:
    from pyrogram.types import AnimatedChatPhoto
except ImportError:
    pytest.skip(
        "This Kurigram version does not provide AnimatedChatPhoto",
        allow_module_level=True,
    )


def make_photo(video_sizes):
    return raw.types.Photo(
        id=1,
        access_hash=2,
        file_reference=b"ref",
        date=0,
        sizes=[],
        dc_id=1,
        video_sizes=video_sizes,
    )


@pytest.mark.asyncio
async def test_animated_chat_photo_ignores_markup_only_video_sizes():
    photo = make_photo(
        [raw.types.VideoSizeEmojiMarkup(emoji_id=1, background_colors=[])]
    )

    parsed_photo = await AnimatedChatPhoto._parse(None, photo)

    assert parsed_photo is None


@pytest.mark.asyncio
async def test_animated_chat_photo_keeps_valid_video_size():
    photo = make_photo([raw.types.VideoSize(type="v", w=320, h=240, size=1024)])

    parsed_photo = await AnimatedChatPhoto._parse(None, photo)

    assert parsed_photo.length == 320
    assert parsed_photo.animation.width == 320
    assert parsed_photo.animation.height == 240
