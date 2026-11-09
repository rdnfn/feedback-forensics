"""Deterministic mock of an OpenAI-compatible chat completions API (for tests).

The mock answers annotation (voting) prompts. Its answers depend only on the
*content* of a request, never on the order in which the two samples are shown:
for each principle the selected sample is determined by hashing the principle
together with each sample's text. Therefore any two correct annotation
pipelines must produce exactly the same votes against this mock, regardless of
how they (randomly) order samples A and B. This makes it possible to rigorously
test that the native FF annotation reproduces the former ICAI pipeline.

To exercise error handling, the mock also deterministically (based on content)
produces: non-applicable/"Both"/"Neither"/unknown vote values, missing votes,
malformed and markdown-wrapped responses, truncated responses
(finish_reason "length") and transient HTTP 500 errors.

With `position_bias=True`, some principles are answered with "A" regardless of
content (to test position debiasing).
"""

import ast
import hashlib
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VOTING_PROMPT_REGEX = re.compile(
    r"Sample A:\n(?P<a>.*)\n\nSample B:\n(?P<b>.*)\n\n"
    r"Given the samples data above, check for each rule below which sample "
    r"should be selected:\n(?P<principles>.*)\n\nAnswer in json format",
    re.DOTALL,
)


def _h(*parts: str) -> int:
    return int(hashlib.md5("\x00".join(parts).encode("utf-8")).hexdigest(), 16)


def pair_key(sample_1: str, sample_2: str) -> str:
    """Order-invariant key of a pair of samples."""
    return "\x00".join(sorted([sample_1, sample_2]))


# Behaviour per (order-invariant) pair, decided by `_h(pair_key) % 100`
FAIL_ONCE_RANGE = range(0, 6)  # HTTP 500 on first attempt (if enabled)
MALFORMED_RANGE = range(6, 10)  # non-JSON text -> all votes invalid
MARKDOWN_RANGE = range(10, 13)  # ```json fenced -> unparsable -> invalid
TRUNCATED_RANGE = range(13, 16)  # cut-off JSON, finish_reason "length"

# Behaviour per (principle, pair), decided by `_h(principle, pair_key) % 20`
PRINCIPLE_NOT_APPLICABLE = 0
PRINCIPLE_BOTH = 1
PRINCIPLE_NEITHER = 2
PRINCIPLE_UNKNOWN_VALUE = 3
PRINCIPLE_MISSING = 4
PRINCIPLE_POSITIONAL = (5, 6)  # only with position_bias=True: always "A"


def principle_behaviour(principle: str, sample_1: str, sample_2: str) -> int:
    return _h(principle, pair_key(sample_1, sample_2)) % 20


def content_choice(principle: str, sample_1: str, sample_2: str) -> str:
    """The sample (text) selected for a principle, independent of order."""
    return max([sample_1, sample_2], key=lambda s: (_h(principle, "|", s), s))


def mock_vote_value(
    principle: str, sample_a: str, sample_b: str, position_bias: bool = False
):
    """Raw vote value for a principle (as the LLM would answer), or MISSING."""
    behaviour = principle_behaviour(principle, sample_a, sample_b)
    if behaviour == PRINCIPLE_NOT_APPLICABLE:
        return "None"
    if behaviour == PRINCIPLE_BOTH:
        return "Both"
    if behaviour == PRINCIPLE_NEITHER:
        return "neither"
    if behaviour == PRINCIPLE_UNKNOWN_VALUE:
        return "C"
    if behaviour == PRINCIPLE_MISSING:
        return MISSING
    if position_bias and behaviour in PRINCIPLE_POSITIONAL:
        return "A"
    return "A" if content_choice(principle, sample_a, sample_b) == sample_a else "B"


MISSING = object()


def mock_response(messages: list[dict], position_bias: bool = False) -> tuple[str, str]:
    """Return (response text, finish_reason) for a voting request."""
    user_content = next(m["content"] for m in messages if m["role"] == "user")
    match = VOTING_PROMPT_REGEX.search(user_content)
    if match is None:
        return "I only answer voting prompts.", "stop"
    sample_a, sample_b = match.group("a"), match.group("b")
    principles = ast.literal_eval(match.group("principles"))

    pair_behaviour = _h(pair_key(sample_a, sample_b)) % 100
    if pair_behaviour in MALFORMED_RANGE:
        return "Sorry, I cannot help with comparing these samples.", "stop"

    votes = {}
    for idx, principle in principles.items():
        value = mock_vote_value(principle, sample_a, sample_b, position_bias)
        if value is not MISSING:
            votes[idx] = value

    # Vary the output format (all variants parsable by the lenient parser)
    style = _h("style", pair_key(sample_a, sample_b)) % 3
    if style == 0:
        text = json.dumps({str(k): v for k, v in votes.items()})
    elif style == 1:
        text = str(votes)  # python dict with int keys and single quotes
    else:
        text = json.dumps({str(k): v for k, v in votes.items()}, indent=2).replace(
            '"None"', "null"
        )

    if pair_behaviour in MARKDOWN_RANGE:
        return f"```json\n{text}\n```", "stop"
    if pair_behaviour in TRUNCATED_RANGE:
        return text[: len(text) // 2], "length"
    return text, "stop"


class MockLLMServer:
    """Threaded HTTP server implementing `POST /v1/chat/completions`."""

    def __init__(
        self,
        position_bias: bool = False,
        fail_first_attempts: bool = False,
        keep_alive: bool = False,
    ):
        """
        Args:
            position_bias: Answer some principles with "A" regardless of content.
            fail_first_attempts: Return HTTP 500 on the first attempt of some
                requests (to test retries).
            keep_alive: Use HTTP/1.1 keep-alive connections (default: HTTP/1.0,
                closing the connection after each request).
        """
        self.position_bias = position_bias
        self.fail_first_attempts = fail_first_attempts
        self.requests: list[dict] = []
        self.num_failed_attempts = 0
        self._attempts: dict[str, int] = {}
        self._lock = threading.Lock()
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1" if keep_alive else "HTTP/1.0"

            def log_message(self, *args):  # silence request logging
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                status, payload = server._handle(body)
                data = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()

    def reset(self):
        with self._lock:
            self.requests = []
            self.num_failed_attempts = 0
            self._attempts = {}

    def _handle(self, body: dict) -> tuple[int, dict]:
        messages = body.get("messages", [])
        request_key = json.dumps(messages, sort_keys=True)
        with self._lock:
            self._attempts[request_key] = self._attempts.get(request_key, 0) + 1
            attempt = self._attempts[request_key]

        if self.fail_first_attempts and attempt == 1:
            user_content = next(
                (m["content"] for m in messages if m["role"] == "user"), ""
            )
            match = VOTING_PROMPT_REGEX.search(user_content)
            if match and _h(pair_key(match["a"], match["b"])) % 100 in FAIL_ONCE_RANGE:
                with self._lock:
                    self.num_failed_attempts += 1
                return 500, {"error": {"message": "Mock server error", "code": 500}}

        with self._lock:
            self.requests.append(body)

        text, finish_reason = mock_response(messages, self.position_bias)
        prompt_tokens = sum(len(m.get("content", "")) for m in messages) // 4
        completion_tokens = len(text) // 4 + 7
        return 200, {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": 0,
            "model": body.get("model"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "completion_tokens_details": {"reasoning_tokens": 7},
                "cost": round((prompt_tokens + 4 * completion_tokens) * 1e-7, 10),
            },
        }
