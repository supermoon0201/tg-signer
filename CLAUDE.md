# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

tg-signer is a Telegram automation tool for scheduled check-ins, message monitoring, forwarding, and auto-replies. It uses the kurigram (Pyrogram fork) Telegram client library, Pydantic for config models, Click for CLI, and OpenAI-compatible APIs for AI-powered image/text recognition tasks.

## Common Commands

```sh
# Install in development mode
pip install -e .
pip install --group dev

# Lint
ruff check

# Run tests
pytest tests/

# Run a single test file
pytest tests/test_core.py -vv

# Run the CLI
tg-signer --help
```

## Architecture

The codebase has three main execution modes, all sharing a common base:

- **Signer** (`core.py:UserSigner`): Scheduled task runner that executes action flows (send text, click buttons, AI image/text recognition, WebApp automation via Playwright) on a cron schedule.
- **Monitor** (`core.py:UserMonitor`): Real-time message listener that matches incoming messages by rules (exact/contains/regex/all) and responds or forwards.
- **Automation** (`automation/engine.py:UserAutomation`): Rule-driven engine with message/timer/startup triggers and handler chains. Recommended over Monitor for new use cases.

### Key modules

- `tg_signer/core.py` — Core classes: `Client`, `BaseUserWorker`, `UserSigner`, `UserMonitor`. Contains all sign-in action logic (button clicking, AI recognition, WebApp/Playwright automation, Turnstile/2Captcha handling).
- `tg_signer/config.py` — Pydantic models for all configuration: `SignConfigV3`, `MonitorConfig`, `AutomationConfig`, action types, match rules. Handles config versioning and migration.
- `tg_signer/cli/signer.py` — Click CLI entry point for signer commands.
- `tg_signer/cli/automation.py` — Click CLI for automation subcommands.
- `tg_signer/cli/monitor.py` — Click CLI for monitor subcommands.
- `tg_signer/automation/` — Automation engine: `engine.py` (executor), `handlers.py` (built-in + plugin handlers), `models.py` (context/state).
- `tg_signer/ai_tools.py` — OpenAI API integration for image recognition, text choice, GIF recognition, calculation problems.
- `tg_signer/sign_record_store.py` — SQLite-based sign record persistence (migrated from JSON).
- `tg_signer/webui/` — Optional NiceGUI-based web interface.

### Data flow

1. CLI parses args → creates a `UserSigner`/`UserMonitor`/`UserAutomation` worker
2. Worker logs in via kurigram `Client` (session stored as `.session` file or in-memory)
3. Worker loads JSON config from `.signer/{signs|monitors|automations}/<task>/config.json`
4. For signer: cron loop → execute action chain per chat → persist record to SQLite
5. For monitor/automation: register message handlers → idle loop

### Config versioning

`SignConfigV3` supports loading from older formats (V1/V2) via `BaseJSONConfig.load()` which returns `(config, from_old)` tuple. When migrated, the new format is written back automatically.

## Environment Variables

- `TG_API_ID` / `TG_API_HASH` — Telegram API credentials (defaults provided)
- `TG_PROXY` — Proxy URL (e.g., `socks5://127.0.0.1:7890`)
- `TG_ACCOUNT` — Account name (default: `my_account`)
- `TG_SESSION_STRING` — Session string for in-memory mode
- `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL` — AI model config
- `TWOCAPTCHA_API_KEY` — 2Captcha service key
- `BARK_URL` / `BARK_SOUND` / `BARK_GROUP` — Bark push notification
- `SERVER_CHAN_SEND_KEY` — Server Chan notification
- `TZ` — Timezone override (fallback: local timezone → Asia/Shanghai)

## Testing

Tests use pytest + pytest-asyncio. No external services required — tests mock Telegram API calls. Run with:

```sh
pytest tests/ -vv -x
```

The `-x` flag stops on first failure (configured in pyproject.toml).
