"""Adapters from existing request types to router calls.

Annotation and generation do not use the router yet: this shows the fit.
"""

from feedback_forensics.models.main import DEFAULT_TEMPERATURE
from feedback_forensics.router.datatypes import Call, Message, Params


def call_from_vote_request(request, config, tag=None) -> Call:
    """The router call of an annotation request.

    The call has the messages and generation settings that annotation sends
    for the request today (live and in batches).

    Args:
        request: `feedback_forensics.annotation.voting._VoteRequest`.
        config: `feedback_forensics.annotation.voting.VotingConfig`.
        tag: Returned with the result (default: the request's id).
    """
    return Call(
        model=config.model,
        messages=tuple(Message(m["role"], m["content"]) for m in request.messages),
        params=Params(
            max_tokens=config.max_output_tokens,
            reasoning_effort=config.reasoning_effort,
            temperature=DEFAULT_TEMPERATURE,
        ),
        tag=request.request_id if tag is None else tag,
    )
