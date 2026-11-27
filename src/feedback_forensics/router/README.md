# Model-call router

Takes a set of LLM calls and gets them answered at the lowest cost: by reusing
stored answers, by ordering calls so that the provider's prompt cache is hit,
and by using batch prices when there is time. It knows nothing about what the
calls are for. Annotation and generation can become callers of it; today they
do not use it yet.

```python
from datetime import timedelta
from feedback_forensics.router import Call, Message, Params, Policy, Router

router = Router(store="path/to/calls", strategy="auto")

keys = router.add(calls, task="annotation", group="openai/gpt-x")  # nothing is sent
report = router.run(Policy(deadline=timedelta(hours=30)))  # plan, send, wait
results = router.collect(keys)                             # key -> Result

plan, layout = router.dry_run(Policy(), check_layout=True)  # no calls sent
```

## Three phases

1. **Queue** (`add`). Queue everything first: only then can the router see what
   calls share, also across callers (e.g. the same user prompt annotated for 36
   models). A call queued twice is answered once. For stages that depend on
   each other (generate, then annotate what was generated), run the router
   twice; calls have no dependencies between them.
2. **Run** (`run`, or `arun` inside an event loop).
   - *Resolve*: drop duplicates, take stored answers.
   - *Plan* (pure: same input, same plan; no network, no disk). Per model, the
     prompts are sorted as text and the common prefixes of neighbours give a
     prefix tree; callers do not mark shared parts. A prefix is kept if two or
     more calls share it and it is long enough to cache for the model; it ends
     at a line break (or at the end of a message). Each call gets at most the
     model's maximum number of cache breakpoints, those with the largest
     saving. A strategy turns the tree into a plan.
   - *Execute*: send the steps, stage by stage, within the concurrency cap and
     the budget. Each answer is written to the store as it arrives.
3. **Collect** (`collect`). Every key gets a `Result`: the answer, or the reason
   there is none (`error`).

`run` is resumable. If it is interrupted, or batches are still pending at the
deadline (`RunReport.pending`), run the same script again: stored answers are
not requested again, and batches under way are collected, never submitted a
second time. One thing cannot be recovered: a live call that was in flight
when the run was interrupted has no answer, so it is sent again
(`RunReport.interrupted` counts them; the provider may have billed them).

### Ordering on the live lane

A cache entry expires five minutes after its last use, and a call can only read
an entry that an earlier call has finished writing. So for each shared prefix
one call goes first and the others wait for it (`Step.after`), then run in
parallel. Calls are started close to their place in the plan (tree order), so
calls sharing a prefix run close together. If the first call fails, one of the
waiting calls takes its place.

## Types

Frozen dataclasses in `datatypes.py`:

| Type | What it holds |
| --- | --- |
| `Call` | `model`, `messages` (`Message(role, text)`), `params` (`Params`: max tokens, reasoning effort, temperature), `tag` (opaque, returned untouched) and `key`: a hash of model, params and messages, independent of the tag and of where the prompt is cut for caching. |
| `Policy` | `deadline` (None: results are needed now, no batches), `budget_usd`, `max_concurrency`, `retries`, `accept` (the caller's check that an answer is usable). |
| `ModelProfile` | Prices and caching behaviour of a model. The planner never looks at the model name. See `profiles.py`. |
| `PrefixTree`, `PrefixNode` | Shared prefixes of one model's calls: `end` (character offset), estimated `tokens`, `keys` of the calls below, `children`. |
| `Plan`, `Stage`, `Step` | Stages run one after another. A step has the call, its `lane` ("live" or "batch"), `breakpoints` (character offsets into the prompt), `after`, `cache_ttl` and its estimated cost. `Plan.estimate` has the cost as given (all live, no caching) and as planned. |
| `Result` | `key`, `tag`, `text`, `finish_reason`, `error`, `usage` (input, cached, cache-written, output and reasoning tokens, cost), `route` ("store", "live" or "batch"), `raw`. |
| `RunReport` | Requests sent, answers reused from the store, tokens, share of input read from cache, cost against the estimate, calls given up, calls pending in batches, problems. |

Model names follow the repository (`openrouter/<provider>/<model>`). Character
offsets count characters of the message texts joined together.

## Strategies

| Name | Plan |
| --- | --- |
| `live_ordered` | Live lane, tree order, all cache layers. |
| `batch` | Everything in batches, assuming no cache hits (no breakpoints). |
| `warm_then_batch` | Stage 1: one live call writes the fixed prefix with a 1-hour lifetime. Stage 2: batches with a breakpoint on that prefix. |
| `batch_chain` | Stage 1: a small batch writes the prefix. Stage 2: the main batches. |
| `auto` | Estimates every registered strategy the policy allows and picks the cheapest that meets the deadline. |

Whether a prefix written live (or by an earlier batch) is read by batch requests
is not known. After a run that sent batch requests to read a prefix, the share
that did is recorded in the model profile (`observed_hit_rate`, kept in the
store's `.router/profiles.json`). Until it is measured, the estimate assumes a
rate at which writing the prefix does not pay off, so `auto` chooses `batch`
over `warm_then_batch` and `batch_chain`. To measure it, run one of the two
explicitly, on a small set.

If a run with stages is continued by a later run, the prefix written by stage 1
may have expired: stage 2 is then sent without breakpoints (at the plain batch
price) instead of having every request pay to write the prefix again.

### Adding a strategy

A strategy is one function from the prefix tree (of one model's calls), the
profiles and the policy to a plan. Decide lane, breakpoints and stage per call;
`finish_plan` derives `after` and the estimate:

```python
from feedback_forensics.router.profiles import get_profile
from feedback_forensics.router.strategies import (
    Draft, choose_breakpoints, finish_plan, strategy,
)

@strategy("live_then_batch")
def live_then_batch(tree, profiles, policy):
    """The first ten calls live, the others in batches."""
    profile = get_profile(profiles, tree.model)
    nodes = choose_breakpoints(tree, profile)
    live = [Draft(call, "live", nodes[call.key]) for call in tree.calls[:10]]
    rest = [Draft(call, "batch") for call in tree.calls[10:]]
    return finish_plan("live_then_batch", tree, profile, policy, [live, rest])
```

Raise `ValueError` if the strategy cannot be used for a policy or model. Once
registered, it can be named in `Router(strategy=...)` and is considered by
`auto`. The assumptions behind the estimates (output tokens per call, duration
of a live call, time until a batch ends) are constants in `strategies.py`.

## Profiles

`profiles.py` has the profile of `anthropic/claude-haiku-5.5` as measured on
OpenRouter in November 2026, and `fetch_profiles(models)`, which builds
profiles from OpenRouter's public model list (prices, cache prices, batch
variant). What the list does not say (minimum cacheable length, number of
breakpoints, characters per token) comes from `KNOWN_CACHING`; a model that is
not in there gets no breakpoints. Pass profiles with `Router(profiles=...)`.

## Transports

`transports.py` has one small interface (`Transport`: `send` for a live call;
`submit`, `poll` and `recover` for batches) and its OpenRouter implementation:
live calls go directly over HTTP to chat completions, so that cache breakpoints
are sent as written; batches go through `feedback_forensics.models.openrouter`
(`submit_batch`, `poll_batch`, `recover_batch`), which keeps the guarantees of
the batch path. `fakes.py` has a fake provider with a prompt cache and a batch
API, and an event loop with virtual time, for tests.

Guarantees on the batch lane (as for annotation batches):

- A batch is recorded before it is submitted and is never submitted twice. If
  the response to a submission is lost, the batch is found again in the
  provider's batch list.
- While calls are in a batch, they are not sent again, on any lane.
- Results that match no queued call, or cannot be read, are saved as received
  (`.router/unmatched/<batch id>.json`).
- Requests that failed within a completed batch are repeated on the live lane.
  If a batch fails, expires or is cancelled, its results are stored and the
  remaining calls are reported, not sent live at the full price.
- A batch the provider does not know is kept, and its calls stay pending,
  until the user removes it from `.router/batches.json`.

## Store

Raw answers only, never parsed. The layout can be uploaded as part of a dataset
and read without this code; `README.md` in the store explains every field and
how keys are computed.

```
calls/
├── README.md
├── pieces/part-000.jsonl      {hash, text}: each distinct piece of prompt text, once
├── annotation/<annotator>/<annotated model>.jsonl    one line per call
└── generation/<model>.jsonl
```

The caller says where a call belongs with `add(calls, task=..., group=...)`:
the file is `<task>/<model>.jsonl`, or `<task>/<model>/<group>.jsonl` with a
group. For annotation the task is "annotation" and the group the annotated
model (the model of the call is the annotator); for generation the task is
"generation", without group.

A line has the key, task, group, model, params, the prompt as hashes of its
pieces (cut where messages end and where the planner placed breakpoints, so
text shared by many calls is stored once), tag, text, finish reason, usage,
route, batch id, provider, creation time and origin ("router" or "migrated").
No API keys. The provider's raw response is not stored.

The router's working state is in `calls/.router/` (pending batches, the plan of
an unfinished run, a log of the state of every call before and after sending).
It holds no answers and can be left out of uploads.

`migrate.py` converts generation files written by `ff_modelgen` into store
records with origin "migrated" (`scripts/migrate_generations_to_call_store.py`).

## Layout check

`dry_run(policy, check_layout=True)` also returns a report of text that repeats
across calls but sits behind text unique to each call, where no cache can reach
it: for each such run of paragraphs, the number of calls, where it sits and the
estimated saving if it were moved earlier. It only reports, as suggestions:
some repeats cannot be moved (e.g. code from the user prompt that several
responses quote).
