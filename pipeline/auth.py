"""TradeStation OAuth 2.0 token manager with thread-safe refresh."""

from __future__ import annotations

import json
import logging
import secrets
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import requests

from pipeline.config import TradeStationConfig

logger = logging.getLogger(__name__)

AUTH_URL = "https://signin.tradestation.com/authorize"
TOKEN_URL = "https://signin.tradestation.com/oauth/token"


class TradeStationAuth:
    """Manages OAuth 2.0 tokens for the TradeStation API.

    Thread-safe: multiple callers can request tokens concurrently.
    Token is refreshed only once even under contention (double-check locking).
    """

    def __init__(self, config: TradeStationConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._tokens: dict = {}
        self._load_tokens()

    def get_access_token(self, skew_sec: int = 180) -> str:
        """Return a valid access token, refreshing if necessary."""
        if self._token_valid(skew_sec):
            return self._tokens["access_token"]

        with self._lock:
            if self._token_valid(skew_sec):
                return self._tokens["access_token"]

            refresh_token = self._tokens.get("refresh_token")
            if refresh_token:
                logger.info("Refreshing TradeStation access token...")
                try:
                    new_tokens = self._refresh(refresh_token)
                    self._tokens.clear()
                    self._tokens.update(new_tokens)
                    self._save_tokens()
                    return self._tokens["access_token"]
                except Exception:
                    logger.warning("Token refresh failed - starting interactive auth")

            logger.info("Starting interactive TradeStation auth...")
            new_tokens = self._interactive_auth()
            self._tokens.clear()
            self._tokens.update(new_tokens)
            self._save_tokens()
            return self._tokens["access_token"]

    def _token_valid(self, skew_sec: int) -> bool:
        exp = self._tokens.get("expires_at", 0)
        return bool(
            self._tokens.get("access_token")
            and time.time() < exp - skew_sec
        )

    def _refresh(self, refresh_token: str) -> dict:
        resp = requests.post(
            TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "refresh_token",
                "client_id": self._config.client_id,
                "client_secret": self._config.client_secret,
                "refresh_token": refresh_token,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        data["expires_at"] = time.time() + data.get("expires_in", 1200)
        return data

    def _interactive_auth(self) -> dict:
        """Open the browser for the user to authorize, then exchange the code."""
        parsed = urllib.parse.urlparse(self._config.redirect_uri)
        port = parsed.port or 8080

        auth_code: list[str | None] = [None]

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                auth_code[0] = qs.get("code", [None])[0]
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(b"<h2>Authorization complete. You may close this tab.</h2>")

            def log_message(self, *_args: object) -> None:
                pass

        state = secrets.token_urlsafe(16)
        params = {
            "response_type": "code",
            "client_id": self._config.client_id,
            "redirect_uri": self._config.redirect_uri,
            "audience": "https://api.tradestation.com",
            "scope": self._config.scopes,
            "state": state,
        }
        url = f"{AUTH_URL}?{urllib.parse.urlencode(params)}"
        webbrowser.open(url)
        logger.info("Opened browser for TradeStation authorization")

        server = HTTPServer(("", port), _Handler)
        server.timeout = 120
        while auth_code[0] is None:
            server.handle_request()
        server.server_close()

        if not auth_code[0]:
            raise RuntimeError("Did not receive authorization code")

        resp = requests.post(
            TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "authorization_code",
                "client_id": self._config.client_id,
                "client_secret": self._config.client_secret,
                "redirect_uri": self._config.redirect_uri,
                "code": auth_code[0],
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        data["expires_at"] = time.time() + data.get("expires_in", 1200)
        return data

    def _token_path(self) -> Path:
        return Path(self._config.token_file)

    def _load_tokens(self) -> None:
        p = self._token_path()
        if p.exists():
            try:
                self._tokens = json.loads(p.read_text(encoding="utf-8"))
                logger.debug("Loaded tokens from %s", p)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not load token file: %s", exc)

    def _save_tokens(self) -> None:
        p = self._token_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self._tokens, indent=2), encoding="utf-8")
        logger.debug("Saved tokens to %s", p)
