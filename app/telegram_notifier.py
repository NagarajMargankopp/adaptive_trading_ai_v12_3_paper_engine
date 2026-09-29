from __future__ import annotations

import os
from typing import Any

import requests


TELEGRAM_API = "https://api.telegram.org"


class TelegramNotifier:
    def __init__(self) -> None:
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    @property
    def enabled(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def send(self, message: str) -> bool:
        if not self.enabled:
            return False

        url = f"{TELEGRAM_API}/bot{self.bot_token}/sendMessage"

        try:
            response = requests.post(
                url,
                json={
                    "chat_id": self.chat_id,
                    "text": message,
                },
                timeout=20,
            )
            response.raise_for_status()

            payload: dict[str, Any] = response.json()
            return bool(payload.get("ok"))

        except Exception as exc:
            print(
                f"[TELEGRAM] Send failed: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return False
