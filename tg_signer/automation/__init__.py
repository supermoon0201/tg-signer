"""Automation engine and handlers."""

__all__ = ["UserAutomation"]


def __getattr__(name: str):
    if name == "UserAutomation":
        from .engine import UserAutomation

        return UserAutomation
    raise AttributeError(name)
