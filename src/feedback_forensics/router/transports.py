"""Transports: how requests reach the provider. One small interface for both
lanes, so that tests can substitute a fake (`feedback_forensics.router.fakes`).

- Live: direct HTTP to OpenRouter's chat completions, because cache breakpoints
  have to be sent as written (LangChain does not pass them on).
- Batch: the functions of `feedback_forensics.models.openrouter`, which give the
  guarantees of the batch path (a batch is tracked before it is submitted, never
  submitted twice, and found again if the response to its submission is lost).
"""

import asyncio
import os
from dataclasses import dataclass
from typing import Callable, Protocol

from feedback_forensics.models import main as models_main
from feedback_forensics.models import openrouter
from feedback_forensics.router.profiles import OPENROUTER_PREFIX, model_id

# HTTP status codes of failed requests worth repeating, as for annotation
# (`feedback_forensics.annotation.voting.RETRYABLE_HTTP_CODES`), and 5xx
RETRYABLE_HTTP_CODES = (403, 408, 409, 420, 429, 444, 498, 499)
# The API key or account is rejected: all further requests would be too
ACCOUNT_ERROR_HTTP_CODES = (401, 402)


class TransportError(RuntimeError):
    """A request failed. `retryable`: repeating it may succeed."""

    def __init__(self, message: str, status_code=None, retryable: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class AccountError(TransportError):
    """The provider rejected the API key or account (e.g. no credits left):
    further requests would be rejected too."""


class BatchNotFound(TransportError):
    """A batch is not known to the provider, long after its submission. It may
    be running (and billed) nonetheless, so its state is to be kept."""


@dataclass(frozen=True)
class Request:
    """A request as sent: `key` of the call (the custom id in a batch), model
    and body (see `feedback_forensics.router.wire.build_body`)."""

    key: str
    model: str
    body: dict


class Transport(Protocol):
    async def send(self, request: Request) -> dict:
        """Send a live request, returns the chat completion.

        Raises:
            TransportError: The request failed (`AccountError`: stop sending).
        """

    async def submit(
        self,
        model: str,
        requests: list[Request],
        entry: dict,
        save: Callable[[], None],
        discard: Callable[[], None],
    ) -> str:
        """Submit requests as one batch, tracked in `entry` (set here:
        `batch_id`, `num_requests`, `submitted_at`), returns the batch id.

        `save()` persists the entry: it is called before submitting (without
        batch id) and after (with it). `discard()` removes it, called if the
        batch was definitely not created.
        """

    async def poll(self, entry: dict) -> tuple[dict | None, str]:
        """(batch, status) of a submitted batch, with batch None while pending.
        A batch that ended has `results` (as OpenRouter's batch API, read with
        `feedback_forensics.models.openrouter.get_result_body`); its status is
        "completed" or says why it did not complete.

        Raises:
            BatchNotFound: The batch is not known to the provider.
        """

    async def recover(
        self, model: str, entry: dict, exclude_ids: list[str]
    ) -> tuple[bool, str | None]:
        """Find the batch of an entry without batch id (the response to its
        submission was lost). Returns (True, id) if found, (True, None) if no
        such batch exists, (False, None) if that is not known yet."""


class OpenRouterTransport:
    """OpenRouter: chat completions over HTTP, batches via
    `feedback_forensics.models.openrouter`. Needs `OPENROUTER_API_KEY`."""

    def __init__(self, max_connections: int = 1000):
        self.max_connections = max_connections
        self._client = None
        self._http_errors: tuple = ()

    def _get_client(self):
        # Created on first use: bound to the running event loop, with the
        # repository's timeouts, TCP keepalive and proxy handling
        if self._client is None:
            self._client = models_main.create_async_http_client(
                "openrouter/", max_connections=self.max_connections
            )
            http = models_main._get_http_library(type(self._client))
            self._http_errors = (http.TransportError, asyncio.TimeoutError)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def send(self, request: Request) -> dict:
        if not request.model.startswith(OPENROUTER_PREFIX):
            raise ValueError(
                "The OpenRouter transport only sends 'openrouter/' models, got "
                f"'{request.model}'."
            )
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise AccountError("OPENROUTER_API_KEY is not set.")
        client = self._get_client()
        try:
            response = await client.post(
                models_main.OPENROUTER_API_BASE + "/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    **models_main.OPENROUTER_HEADERS,
                },
                json={"model": model_id(request.model), **request.body},
            )
        except self._http_errors as e:
            raise TransportError(f"{type(e).__name__}: {e}", retryable=True) from e
        status, text = response.status_code, response.text
        if status == 200:
            try:
                body = response.json()
            except ValueError as e:
                raise TransportError(f"unreadable response: {e}", 200, True) from e
            error = body.get("error") if isinstance(body, dict) else None
            if not error and isinstance(body, dict):
                return body
            # OpenRouter reports some provider errors with HTTP status 200
            code = error.get("code") if isinstance(error, dict) else None
            status = code if isinstance(code, int) else 502
        raise _http_error(status, text)

    async def submit(self, model, requests, entry, save, discard) -> str:
        try:
            batch = await asyncio.to_thread(
                openrouter.submit_batch,
                model,
                [{"custom_id": r.key, "body": r.body} for r in requests],
                entry,
                save,
                discard,
            )
        except Exception as e:  # pylint: disable=broad-except
            raise _batch_error(e) from e
        return batch["id"]

    async def poll(self, entry):
        try:
            return await asyncio.to_thread(
                openrouter.poll_batch, entry["batch_id"], entry["submitted_at"]
            )
        except Exception as e:  # pylint: disable=broad-except
            raise _batch_error(e) from e

    async def recover(self, model, entry, exclude_ids):
        try:
            return await asyncio.to_thread(
                openrouter.recover_batch,
                model,
                entry,
                entry["custom_ids"],
                exclude_ids,
            )
        except Exception as e:  # pylint: disable=broad-except
            raise _batch_error(e) from e


def _http_error(status: int, text: str) -> TransportError:
    error = openrouter.get_openrouter_error(status, text)
    if (
        isinstance(error, openrouter.OpenRouterAccountError)
        or status in ACCOUNT_ERROR_HTTP_CODES
    ):
        return AccountError(str(error), status)
    retryable = status in RETRYABLE_HTTP_CODES or status // 100 == 5
    return TransportError(str(error), status, retryable)


def _batch_error(error: Exception) -> Exception:
    """Errors of the batch API as transport errors (others unchanged)."""
    import requests

    if isinstance(error, openrouter.BatchNotFoundError):
        return BatchNotFound(str(error), 404)
    if isinstance(error, openrouter.OpenRouterAccountError):
        return AccountError(str(error), error.status_code)
    if isinstance(error, openrouter.OpenRouterError):
        retryable = error.status_code == 429 or error.status_code >= 500
        return TransportError(str(error), error.status_code, retryable)
    if isinstance(error, requests.RequestException):
        return TransportError(f"{type(error).__name__}: {error}", retryable=True)
    return error
