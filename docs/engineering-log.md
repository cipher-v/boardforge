# BoardForge engineering log

This is the running implementation journal for substantial BoardForge tasks.
Future work should add a dated section describing the request, problems found,
decisions, changes, verification, and anything still requiring human action.

The goal is to preserve both the technical reasoning and a plain-language
explanation of why each change was made. It is not a changelog: routine file
lists belong in Git history; this document records engineering evidence and
trade-offs.

## 2026-11-25 - v4 live-agent provider failure investigation

### Root cause and artifact evidence

No external request was made during this investigation. The preserved v4
artifact shows that the runner constructed a `vertex-gemini` provider in
`global` with model `openai/gpt-oss-120b`. Vertex returned an authenticated
404 before inference: the publisher model
`publishers/openai/models/gpt-oss-120b` was not found or accessible. The v3
artifact used `vertex-gemini` with `gemini-3.8-flash` in the same project and
location successfully. Both first planning reservations were identical (2,507
tokens and $0.00418425), which also rules against a changed prompt or tool
declaration as the cause of this pre-inference failure.

The live runner forced `BOARDFORGE_RELEVANCE_PROVIDER=vertex` but did not force
the corresponding model. The interactive PowerShell process had loaded the
local `.env`, whose generic relevance-model setting belongs to the manually
selectable Groq configuration. The earlier Codex process had not imported that
setting and therefore used the Vertex adapter's Gemini default. This was a
latent environment-coupling defect in the evaluation runner, exposed after the
recent reliability work; it was not caused by candidate validation, agent
finalization, or ranking changes. No search or verifier code ran. Authentication
was not the failure, and request parsing, SDK response parsing, and native
function declarations had not yet become relevant.

The artifact contains the complete chained exception message and one failed
ledger row, but no Python stack trace and no response metadata. Its provider
trace records one attempt, zero input/output/thinking tokens, zero estimated
provider cost, and zero reported inference latency. The ledger row retains the
$0.00418425 reservation for a failed call by design: cumulative protection
counts completed calls at estimated cost and all non-completed calls at their
reserved cost. It is conservative budget accounting, not evidence of billable
provider usage.

### Corrections

- The frozen live runner now explicitly pins the v3 provider/model pair
  (`vertex-gemini` / `gemini-3.8-flash`) and validates it before an agent budget
  reservation can be created. Shell and `.env` defaults can no longer silently
  cross-wire a Groq model into Vertex.
- Vertex request exceptions now preserve elapsed provider latency plus bounded,
  credential-free phase, exception type, status, model, and location metadata.
  This corrects the misleading zero-latency diagnostic. Full stack traces are
  intentionally not serialized into durable artifacts; Python exception
  chaining remains available to local logs/tests.
- Live reports now distinguish token-based provider estimated cost from the
  conservative ledger-accounted cost. Failed reservations remain retained, so
  the safety budget is not weakened.
- Regression tests cover shell-model contamination, failed-reservation cost
  reporting, and request-failure latency/metadata.

During full-suite verification, the first version of the shell-contamination
test exposed its own process-environment leak: live shortlist limits remained
set for later verifier batching tests. The configuration helper is now
side-effect-free; only the actual runner applies the returned settings, and the
test uses pytest-managed environment restoration.

Verification completed with **42/42 focused provider and intelligent-agent
tests**, **40/40 deterministic mocked-agent scenarios**, and **211/211 full
tests**, including the isolated `boardforge_test` PostgreSQL/pgvector test. The
only warning is the existing Starlette/httpx deprecation. `git diff --check`
also passed.

The root cause is fully established from existing evidence. A paid rerun is not
needed to diagnose or validate this fix; any future live scenario should occur
only when its product behavior—not this incident investigation—requires a
separately authorized validation.

## 2026-11-25 - Agent reliability closure after v3 review

### Manual changes reviewed

The manual verifier change was correct in its main design: provider reasoning is
no longer rejected solely for exceeding 160 characters, while the resolved
judgment still has a bounded explanation. Evidence IDs, relevance enums,
candidate isolation, source-hash cache keys, deterministic eligibility, and
exact source-segment resolution remain strict. The fallback cap was also
correctly reduced from 500 to 160 characters.

The manual agent finalization was likewise directionally correct. It preserved
the real `iteration_limit`, constructed the final response deterministically,
and did not spend another model call merely to summarize a completed turn. The
partial `verification_incomplete` classification correctly considered provider
candidate-validation failures and budget, deadline, and quota skips.

### Bugs found and corrected

1. `verification_incomplete` had no grounded-message branch. A fast-path search
   could therefore carry the correct stop reason while telling the user that no
   suitable jobs existed. Messages now independently report execution limits,
   verification failures/incompleteness, and unresolved eligibility. An
   iteration limit remains the actual stop reason even when the latest
   verification outcome is incomplete.
2. The evidence-support heuristic included provider reasoning tokens. Removing
   the reasoning length cap made that unsafe: a verbose explanation could
   manufacture lexical overlap for unrelated source evidence. Provider
   reasoning is no longer evidence. Direct judgments require substantive
   responsibility evidence that overlaps the source title or requested intent;
   adjacent and irrelevant judgments require substantive duties. Instruction
   text, benefits, optional qualifications, and company “we build” boilerplate
   cannot establish direct relevance. Explicit no-vacancy evidence remains
   valid for an irrelevant judgment.
3. Prefix-truncating a long provider explanation could remove a later qualifier
   and produce a misleading sentence. Full source/evidence validation now runs
   first. If a valid explanation exceeds 160 characters, BoardForge stores a
   deterministic statement of the validated relevance label and evidence IDs
   rather than a partial provider sentence.
4. Invalid evidence IDs, omitted candidates, duplicate candidate judgments, and
   other candidate-local fallbacks were conservative in ranking but were not all
   exposed in `candidate_validation_failures`. They are now observable, allowing
   orchestration to distinguish incomplete verification from a genuine absence
   of recommendations.
5. A later incomplete refinement could overwrite a more useful earlier result.
   Agent state now keeps the latest search for planning and a separate best
   completed search for user-facing output. A verifier failure or empty partial
   refinement cannot erase earlier viable candidates; the final message still
   reports the latest incomplete outcome.
6. The live runner previously had only a stop-reason check. It now classifies
   verified recommendations, genuine empty results, no confident results,
   unresolved eligibility, verification incompleteness, provider failures, and
   execution limits separately. `verification_incomplete` cannot be counted as
   a successful terminal, while an explicit, fully verified empty result can.
7. One deadline regression used only a 100 ms scheduling margin and became
   flaky under the complete suite. Its timing window was widened so it reliably
   exercises a completed first search followed by a rejected refinement.

Because validation semantics became stricter, the relevance cache format was
bumped from version 2 to version 3. Judgments accepted under the old
reasoning-assisted evidence heuristic will not be reused.

### Verification

- Focused relevance-verifier and intelligent-agent suite: **57 passed**.
- Existing deterministic/mock agent evaluation: **40/40 exact sequences**;
  precision, recall, and F1 all 1.0.
- Complete suite with isolated `boardforge_test` PostgreSQL/pgvector enabled:
  **208 passed** with one existing Starlette/httpx deprecation warning.
- No provider call, live-agent rerun, embedding regeneration, discovery,
  deployment, frozen-label change, or frontend work occurred.

The backend milestone is ready for frontend integration subject to one final
separately authorized smoke validation. Use a single cold/warm-explicit run,
zero retries, the frozen `boardforge_holdout_eval` database and isolated profile,
and a **$0.02 all-inclusive incremental ceiling**. Do not request a warm replay
in the same command: v3 query vectors can be reused, but relevance cache version
3 intentionally starts cold after the validation-policy change.

## 2026-11-25 - Final optimized hybrid-agent live validation

### Preflight and authorization

The complete suite passed 198 tests, including the isolated PostgreSQL/pgvector
integration test. The first database diagnostic deliberately blocked execution:
the default local `boardforge` database contained only four jobs and was not the
frozen evaluation corpus. No paid call had occurred. The local database catalog
was inspected and `boardforge_holdout_eval` was identified by the exact prior
state: 100 active jobs, the isolated `agent-live-eval-20261125` profile, 100
compatible document embeddings, and the previous $0.01491375 usage baseline.

The v1 and v2 artifacts were preserved. The new trace was written incrementally
to `results/agent-live-smoke-vertex-v3.json`. The $0.02 incremental authorization
created an absolute ledger ceiling of $0.03491375. Agent, verifier, and embedding
retries were all zero; discovery, source synchronization, and document backfill
were disabled.

### Result

The primary execution completed in **33.19 seconds**, versus 45.11 seconds in
the previous run (11.92 seconds / 26.4% faster). Authenticated context loaded
locally in **3.43 ms**, replacing the previous 22.19-second context-read model
decision. The primary search fell from 13.15 seconds to **8.01 seconds**, and two
conditional refinements that previously could not finish completed in 4.79 and
4.37 seconds.

The result remains honestly classified **incomplete**, but not because of a
provider or tool failure. Gemini used all three allowed planning iterations to
issue materially different indexed searches, so the turn ended at
`iteration_limit` without an explicit terminal decision. All searches completed,
there were no duplicates or unnecessary calls, and the final five uncertain
results were preserved.

### Calls, caches, and cost

- Agent Gemini: 3 calls; 6,551 input, 244 output, 666 thinking tokens;
  $0.00832575.
- Verifier Gemini: 3 calls; 6,159 input and 317 output tokens; $0.00580800.
- Vertex query embeddings: 3 calls, 122 tokens; $0.00001830.
- Total: **14,059 tokens and $0.01415205**, with zero retries and zero provider
  failures. $0.00584795 of the authorization remained.
- All 100 document embeddings were reused. The three distinct search queries
  were query-cache misses and were persisted. Relevance cache hits were zero.
- Each verifier response failed only the individual candidate reasoning-length
  limit (`too_long`). Candidate-isolated validation preserved the batch and
  conservatively returned `uncertain`; invalid judgments were not cached.
- A warm replay was requested but correctly skipped because the remaining
  $0.00584795 was below the declared $0.006 safe admission floor. No warm result
  is mislabeled as a cold benchmark, and no additional paid request was made.

### Correctness

Tool selection was correct and limited to indexed search. The profile,
preferences, and feedback were unchanged. Known five-year requirements remained
within the saved ceiling, while unknown experience eligibility stayed visibly
unverified and was never overridden. No hard constraint or rejected-memory rule
was violated. There were zero verified recommendations and zero alternatives;
therefore recommendation evidence grounding is not claimed. This may reflect a
genuine corpus gap, but the agent did not reach a conclusive no-suitable-jobs
terminal state.

The readable comparison is
`results/agent-live-smoke-vertex-v3-report.md`. The next engineering work should
reserve a final decision after bounded refinements and align the verifier's live
reasoning output with candidate-level limits without weakening evidence or
eligibility validation. No further live evaluation should be run until those
changes are implemented and separately authorized.

## 2026-11-25 - Hybrid agent orchestration and deadline admission

### Why this milestone was needed

The corrected live trace was valid but incomplete: 45.11 seconds produced no
verified recommendation. The largest avoidable item was a 22.19-second Gemini
decision whose only action was to read context that BoardForge could load
deterministically. The first indexed search then used 13.15 seconds, including
6.11 seconds for a query embedding and 6.62 seconds for relevance verification.
Gemini requested a justified refinement, but only about 1.8 seconds remained;
the refinement embedding started and was cancelled by the outer tool timeout.

### Execution design

- The authenticated profile, durable preferences, and recent feedback are now
  loaded locally before the first model request. A `context_preloaded` trace
  event records this without exposing a redundant context-read tool to Gemini.
- A conservative classifier sends a request directly to indexed search only
  when the deterministic planner sees exactly one search and the text has no
  conditional, alternative, comparison, sequencing, or preference-change cue.
  This path makes zero agent-planning model calls. Complex requests retain
  native Gemini function calling.
- The fast path still passes saved hard constraints and profile-derived soft
  preferences through the existing deterministic search, confidence,
  eligibility, rejected-memory, and relevance-verification layers. It is an
  orchestration shortcut, not a ranking shortcut.
- Complex turns receive the preloaded context in the first planning payload.
  The previous live scenario should therefore need two agent decisions (initial
  search and refinement/stop) rather than three (context read, search,
  refinement), while a straightforward search needs none.

### Deadline and partial-result behavior

- Expensive search, discovery, and synchronization tools are rejected before
  execution when fewer than the configured minimum seconds remain. The trace
  records `insufficient_remaining_time`, and any completed `last_search` remains
  available to the user.
- The same absolute monotonic deadline reaches retrieval and relevance
  verification. An uncached query embedding is not started below its threshold;
  cached vectors remain usable. Likewise, cached relevance judgments are loaded
  first and uncached verification is conservatively left as
  `needs_verification` if insufficient time remains.
- Stopping now distinguishes `no_verified_recommendations` from
  `verification_provider_failure`. Unknown eligibility remains an explicit
  deterministic restriction and cannot be upgraded by orchestration.

### Persistent query cache and connection reuse

Query vectors are cached by normalized semantic query plus embedding model,
dimensions, task type, and cache-format version. SQLite and PostgreSQL both
implement the cache; PostgreSQL migration `008` creates the isolated vector
table. A cache hit updates usage metadata and avoids another paid embedding
request. The Vertex embedding provider now reuses one HTTP client and bounded
semaphore instead of rebuilding a connection pool for every call. The Gemini
SDK client was already lazily initialized and reused.

The production cold reference remains the prior live trace: 45.11 seconds
end-to-end, with 6.11 seconds of cold query-embedding latency. No paid request
was allowed in this milestone, so a new production cold/warm comparison was not
fabricated. The warm paths were verified locally: a normalized repeat query
used the persistent cache with zero additional embedding calls, and the already
measured relevance-cache replay remains 16.04 ms. A final live validation is
still required to measure the combined warm production request.

### Issues encountered and resolutions

- This session's supplied `E:\mnt\data\matcheval` directory did not exist; the
  existing uncommitted repository was verified at `E:\boardforge` and used to
  avoid creating a divergent copy.
- The first pytest attempt used the system interpreter, which lacked project
  dependencies. The project `.venv` was used thereafter.
- The managed sandbox could not create pytest temporary directories beside the
  existing repository. Tests were rerun with a narrowly scoped temp directory.
- No PostgreSQL test URL was configured. A separate `boardforge_test` database
  was created in the existing local pgvector container; the application
  `boardforge` database was not truncated or modified by the integration test.
- One legacy agent test expected Gemini to attempt an unauthorized preference
  update for a simple query. Since simple queries now bypass Gemini, the test
  request was made genuinely conditional so it continued to exercise the
  authorization boundary instead of defeating the fast path.

### Verification and remaining work

- Focused agent, semantic-cache, Vertex-embedding, and relevance regressions:
  **63 passed**.
- Existing deterministic/mock planner evaluation: **40/40 exact sequences**,
  precision/recall/F1 all 1.0.
- Isolated PostgreSQL/pgvector integration, including migration 008 and query
  cache round-trip: **1 passed**.
- Complete suite with PostgreSQL enabled: **198 passed**, with one existing
  Starlette/httpx deprecation warning.
- No provider, embedding, discovery, deployment, or other paid call was made.

The next live validation should use the same frozen scenario and compatible
caches, retries set to zero, and a **$0.02 incremental all-in ceiling**. This is
over three times the previous measured $0.00589545 cost while remaining small
enough to catch accidental call multiplication. It should report cold and warm
turns separately. Remaining bottlenecks are uncached relevance verification,
cold Vertex network/auth latency, and an occasional second planning decision
for a justified conditional refinement.

## 2026-11-25 - Corrected live agent rerun with incremental trace

### Authorization and preflight

The earlier incomplete artifact was preserved at
`results/agent-live-smoke-vertex-v1.json`. A distinct corrected artifact path was
verified absent before the one authorized rerun. The same frozen 100-job corpus,
isolated profile, and 100 cached document embeddings were reused. The existing
ledger baseline was $0.00901830; the fresh hard ceiling was therefore
**$0.04901830**, exactly baseline plus the newly authorized $0.04.

Before any paid request, the exact usage-ledger export query was executed against
PostgreSQL, cache/job counts were checked, the corrected event-checkpoint tests
passed, and the complete suite passed 190 tests. Discovery and source sync were
disabled, document backfill was not available, and all provider retries remained
zero.

The orchestrator now invokes an atomic event sink after every model request,
model decision, tool execution, state transition, and stop condition. Each event
contains exact arguments/results, before/after state, elapsed time, and a current
ledger snapshot. A sink failure stops the turn rather than permitting untraced
paid work.

### Exact execution trace

The corrected run produced this trace:

1. Gemini called `read_user_context(include_feedback=true)`.
2. Gemini called `search_indexed_jobs` with query
   `backend engineer Python distributed systems`, top-K 10, backend/software
   engineering plus a five-year ceiling as mandatory context, Python/FastAPI/
   PostgreSQL/distributed systems/APIs as soft preferences, and cloud-security
   software engineering as the conditional alternative.
3. The indexed search returned three US candidates but zero confident
   recommendations. Ramp Backend Ops had a known five-year requirement;
   Palantir Backend and Roblox Early Career had unknown experience eligibility.
4. Gemini made a materially different refinement:
   `cloud security software engineer backend`, explicitly including USA and the
   five-year ceiling. This was justified because the first result set had zero
   confident recommendations and was not a duplicate.
5. Only about 1.8 seconds remained in the 45-second turn budget. The refinement's
   query embedding began but was cancelled by the tool timeout. The recorded run
   therefore stopped incomplete at the iteration boundary, with no final verified
   recommendation.

Native Vertex function calling was compatible: three agent decisions returned
valid native function calls and all exact payloads were captured. Tool selection
was correct, persistent preferences/profile/feedback were unchanged, no forbidden
tool was attempted, and no duplicate search occurred. The refinement was
semantically justified, but its timed-out execution counts as one unsuccessful
call. Because the recommendation set was empty, grounding, relevance support,
and recommendation-constraint satisfaction are correctly reported false rather
than vacuously true.

### Usage and latency

The fresh run added six ledger reservations:

- three agent Gemini calls;
- one relevance-verification Gemini call;
- one completed query embedding and one cancelled query-embedding reservation;
- zero retries;
- 6,190 accounted tokens; and
- **$0.00589545** incremental estimated cost, including the conservatively
  retained $0.00000615 reservation for the cancelled embedding.

The new ledger total is $0.01491375, leaving $0.03410455 of this run's authorized
increment unused. End-to-end latency was 45,114.74 ms. Agent provider inference
summed to 28,268.93 ms: 22,194.52 ms to decide to read context, 2,940.61 ms to
plan the primary search, and 3,133.80 ms to plan the refinement. The first search
took 13,153.54 ms: query embedding 6,111.29 ms, PostgreSQL retrieval 84.20 ms,
ranking 313.42 ms, and relevance verification 6,623.33 ms (6,595.39 ms provider
inference). The cancelled refinement consumed 1,819.52 ms.

### Reliability findings and post-run fixes

The first relevance response returned a relevance value outside the allowed
literal enum. Candidate-local validation correctly failed closed, but the
transport schema had become too permissive while isolating explanation-length
failures. The transport now constrains candidate reference, relevance enum, and
evidence-ID types while leaving only explanation length for individual semantic
validation. This preserves partial-candidate isolation and restores native enum
guidance.

Cancellation exposed a second issue: `asyncio.CancelledError` left the embedding
reservation in `reserved`. The embedding provider now finalizes cancellation as
`failed` while retaining conservative reserved cost because provider-side usage
is unknown. The live row was reconciled the same way. Tool timeout now becomes
an explicit `tool_timeout` stop condition rather than the misleading
`iteration_limit` label.

The final post-run suite passes **192 tests**, including regressions for atomic
incremental trace events, explicit tool-timeout stopping, cancelled embedding
accounting, relevance-enum schema guidance, and PostgreSQL diagnostics. Only the
existing Starlette deprecation warning remains.

The complete artifact is `results/agent-live-smoke-vertex-v2.json`. The run is
honestly classified **incomplete**: it validates orchestration and safety, but
does not establish successful recommendation delivery. Frontend integration
should wait until the latency budget and verifier-label reliability are validated
in a separately authorized smoke; no additional live call was made here.

## 2026-11-25 - Controlled live Gemini agent evaluation (incomplete)

### Guarded setup

One live scenario was authorized with a hard **$0.04 incremental ceiling**. It
used Vertex ADC, `gemini-3.8-flash` in `global`, the isolated
`boardforge_holdout_eval` PostgreSQL database, an isolated profile ID, 100 frozen
indexed jobs, and all 100 existing `vertex:gemini-embedding-001:768` document
embeddings. The persistent ledger began at zero in this isolated database, so
both LLM and embedding gates received an absolute $0.04 ceiling.

The scenario asked the agent to read saved preferences, search for US backend
roles requiring no more than five years of experience, treat cloud-security
software engineering as a conditional alternative, keep persistent state
unchanged, and use only the existing index. Discovery and trusted-source sync
were disabled at the executor capability boundary. Agent, relevance, and
embedding retries were set to zero. Relevance was limited to one four-candidate
batch; agent reasoning was limited to three iterations and five tool calls.

Before the paid execution, the complete suite passed 188 tests. No Tavily,
external ATS sync, document embedding backfill, holdout rerun, label change, or
deployment occurred.

### Actual usage and recoverable evidence

The scenario made seven recorded provider requests with zero retries:

- five `vertex-gemini` requests;
- two `vertex-embedding` query requests;
- 8,033 input, 382 output, and 427 thinking tokens (8,842 total);
- **$0.00901830** estimated cost; and
- $0.03098170 of the authorized increment remained.

The provider activity spanned 34,409.98 ms from the first reservation to the
last completion. Individual reservation wall times were 10,862.46, 2,556.91,
1,808.03, 6,227.68, 4,192.35, 1,821.86, and 3,273.68 ms. Two query-embedding
reservations prove that indexed search executed at least twice. No document
embedding was regenerated. The isolated preferences/profile and empty feedback
history remained unchanged, and discovery/source sync were structurally
unavailable.

### Reporting failure and honest status

The paid agent execution returned to the runner, but post-processing then queried
`llm_usage_reservations ORDER BY created_at,id`. PostgreSQL correctly rejected
this because the table key is `reservation_id`, not `id`. The exception handler
repeated the same faulty query. Although the minimal artifact was written in the
`finally` block, the in-memory agent result had not yet been checkpointed.

Consequently the exact native function calls and arguments, recommendations,
grounded final message, duplicate/unnecessary-call count, and stop reason cannot
be recovered. They are explicitly null/unscored in the artifact. Search-tool
execution provides partial evidence that native function calling worked, but it
is not sufficient to claim complete compatibility or scenario success. The run
is therefore **incomplete**, and no second live scenario was attempted.

The exporter now orders by `created_at,reservation_id` and checkpoints the full
agent result immediately after `agent.run`, before any diagnostic database
query. A PostgreSQL integration regression exercises this helper. The final
post-live suite again passes **188 tests** with only the existing Starlette
deprecation warning. The recovered machine-readable artifact is
`results/agent-live-smoke-vertex-v1.json`; changes remain uncommitted.

## 2026-11-25 - Intelligent agent orchestration milestone

### Verifier failure isolation

The relevance provider now validates a permissive batch transport envelope
before enforcing semantic limits independently for each candidate. A verbose or
otherwise malformed candidate judgment falls back to `uncertain` without
discarding valid peers from the same paid response. Candidate-level failures are
reported separately from provider/batch failures. Evidence IDs, source-content
resolution, unsupported-evidence rejection, eligibility restrictions, caching,
and budget accounting are unchanged. A regression test sends one valid direct
judgment beside a 161-character explanation and verifies that only the latter
fails closed.

The previous negative-query failure was inspected without changing prompts,
labels, or ranking rules. For `technical recruiter hiring engineers`, Gemini
interpreted descriptions saying that engineering teams were "hiring engineers"
as direct support for the recruiter intent. It therefore upgraded five backend,
business-systems, QA, and crawler-engineering postings. The cited evidence was
real but described the object of hiring, not recruiting as the candidate's
occupation. This is a general subject/object and occupational-intent error in
the semantic judgment; it was documented rather than patched with a
benchmark-specific recruiter rule. The paid holdout was not rerun.

### Orchestration architecture

The existing deterministic `JobAgent`, retrieval, relevance verifier, stores,
discovery queue, sync logic, and fallback planner remain intact. A new bounded
orchestrator sits above them. Vertex Gemini is primary and uses native function
declarations through the shared generation provider. Groq remains manually
selectable with the fixed `openai/gpt-oss-120b` model; there is no automatic
fallback or routing.

Typed contracts cover indexed search, targeted discovery, trusted-source sync,
profile/preferences reads, explicitly authorized preference writes, feedback
reads, and feedback writes. The deterministic executor enforces these rules:

- search must precede discovery, and discovery is denied after sufficient
  verified recommendations;
- preference updates require an exact authorization quote from the current user
  message;
- feedback writes require the stable job ID and verdict in the current message;
- source sync accepts only an exact source already stored in the trusted-source
  registry, never an arbitrary URL;
- identical calls are rejected, tool results are bounded, descriptions remain
  untrusted, and final job summaries use only grounded tool results; and
- there are no code execution, credential access, job invention, or job
  application tools.

Per-turn iteration, attempted-tool, token, provider, per-call timeout, total
timeout, and cumulative-cost gates stop the loop. Every model call uses the
existing persistent atomic ledger. Reservations include serialized native tool
schemas, bounded output, and configured retry capacity; failed calls retain
conservative accounting. Search outputs expose at most ten bounded matches and a
small retrieval summary rather than feeding the complete verifier report back
into the model.

### Reproducible mocked evaluation and traces

`boardforge/data/agent_scenarios_v1.json` freezes twelve mocked scenario
families: straightforward search, conditional preferences, preference changes,
insufficient jobs, discovery, feedback, contradictory constraints, no suitable
jobs, provider failure, budget exhaustion, malicious job text, and repeated
calls. The scorer measures task completion, constraint preservation, appropriate
tool choice, unnecessary calls, stopping, state correctness, provider/tool
calls, tokens, and estimated cost. All requested scenario outcomes and safety
assertions passed. Three intentionally unnecessary/invalid attempts were
observed across adversarial traces and were denied: discovery before search and
the repeated identical search (counted as both repetition and failed call).

Representative deterministic traces were:

1. `read_user_context -> search_indexed_jobs -> sufficient_results` for a
   conditional profile-led search;
2. `search_indexed_jobs -> discover_job_sources -> provider_finished` when the
   index had no suitable result;
3. `update_preferences (authorized) -> search_indexed_jobs ->
   sufficient_results` for an explicit preference change;
4. `search_indexed_jobs -> duplicate_tool_call (denied) -> provider_finished`;
5. `provider_failure -> stop`, with no tool or state mutation; and
6. `budget_cumulative_cost_limit -> stop`, before a provider request.

The preserved 40-case deterministic tool benchmark remains perfect: exact
sequence accuracy and tool-set precision/recall/F1 are all **1.0**. The complete
suite passes **186 tests**, including the isolated PostgreSQL/pgvector test. The
only warning is the existing Starlette `httpx` deprecation notice.

No live LLM, embedding, Tavily, ATS, discovery, deployment, or holdout call was
made. Using the actual serialized system instruction, native tool schemas, and
bounded ten-result shape, a representative three-call Vertex agent turn reserves
1,442 / 1,514 / 2,777 input tokens plus 768 output tokens per call. At the
configured $0.75/M input and $3.75/M output rates, the no-retry estimate is
**$0.01293975**. The conservative all-three-attempt retry capacity is about
**$0.03881925**; a controlled live evaluation should authorize no more than
$0.04 and retain the existing cumulative ledger gate.

## 2026-11-25 - Gemini truncation fix and completed guarded holdout

### Structured-output sizing and failure accounting

The fixed 640-token response cap was replaced by a candidate-count-aware
budget derived from the compact response schema. Candidate references remain
`C001`-style identifiers, evidence is limited to two `E001`-style references,
and reasoning is bounded. The estimator serializes a worst-case valid response,
uses a conservative three JSON bytes per token, adds LOW-thinking and framing
headroom, rounds to a 64-token boundary, and respects a configurable 1,536-token
ceiling. The corrected eight-candidate smoke therefore received 960 output
tokens rather than an unexplained fixed increase.

Vertex responses now preserve finish reasons, response/model identifiers,
latency, usage and estimated cost before JSON/schema validation. `MAX_TOKENS`
is an explicit provider failure, and incomplete JSON or schema-invalid output
cannot be accepted as partial judgments. Failed structured responses are
charged to the persistent usage ledger when measurements are available; if
they are unavailable, the conservative reservation remains. Regression tests
cover truncation, malformed JSON, metadata retention and usage accounting.

After evaluating the holdout, the reasoning bound was increased from 120 to
160 characters because six otherwise structured batches exceeded the original
field limit. This is a general schema robustness correction, not a benchmark
label or prompt-content change. No paid rerun was made, so the results below
faithfully describe the 120-character evaluation configuration; future calls
will use the still-bounded 160-character schema and its recalculated budget.

### Exactly-one corrected smoke

The sole corrected smoke reused the same eight frozen `backend_general`
candidates, ADC, `gemini-3.8-flash`, the `global` Vertex location, LOW thinking,
zero retries and the existing isolated evaluation database. It completed in
one HTTP attempt with finish reason `STOP`:

- structured-output compliance: **yes**;
- evidence integrity: **8/8 judgments valid**, with no fabricated IDs;
- judgments: **6 direct, 1 adjacent, 1 irrelevant**;
- input / output / thinking tokens: **12,348 / 565 / 0**;
- provider / end-to-end latency: **7,506.59 / 8,142.85 ms**; and
- measured estimated cost: **$0.01137975**.

The direct candidates were the two Palantir backend roles, Perplexity's search
crawler and backend internship, Ramp's operations backend role, and
Cloudflare's fintech backend role. Roblox's early-career software role was an
adjacent alternative; its engine/graphics role was irrelevant. Resolved
evidence text came only from the original descriptions.

### Reservation decision and completed holdout

The schema-worst-case plan would have cost about $0.546749 for the remaining
uncached work and exceeded the available cumulative ceiling by roughly $0.008.
The successful smoke supplied measured output requirements: 565 tokens for
eight judgments. Reserving a 25% output margin, rounded per batch to 64-token
boundaries, produced a $0.513698 holdout reservation. Together with the
conservatively retained failed-smoke reservation and the successful smoke,
that plan fit below the authorized **$0.60 cumulative ceiling**, so evaluation
was permitted. Atomic reservations remained the hard stop throughout.

The frozen 100-job, 36-query, AI-audited (not human-verified) holdout completed
all queries with identical pre-verification pools. It made **57 calls / 57 HTTP
attempts / 0 retries**, used **486,532 input, 20,403 output and 0 thinking
tokens**, and cost **$0.44141025**. Eight judgments were served from cache.

For the 32 queries with relevant jobs, the verified system achieved:

- candidate recall **0.9437**, Precision@10 **0.3281**, Recall@10 **0.5563**;
- NDCG@10 **0.6243** and graded NDCG@10 **0.6438**;
- recommended precision **0.9583**, recommended recall **0.1991**;
- recommendation coverage **0.3404**, abstention **0.1986**;
- recommended irrelevant rate **0.0417**; and
- direct recommendation precision **0.6604** across all queries.

The verifier recovered **13/38** previously missed relevant recommendations,
made three incorrect upgrades and one incorrect downgrade. It produced two
false recommendations across two positive queries. For the four queries with
no relevant jobs, one query received five false recommendations; the other
three correctly had none. Hard-constraint violations remained **zero**.

Six provider results failed strict schema validation because reasoning exceeded
the then-120-character bound. They affected `streaming_pipelines`,
`data_plumbing_paraphrase`, `qa_automation`, `technical_support`,
`legal_counsel`, and `no_cobol_mainframe`; all 31 affected candidates failed
closed to uncertain. There were no `MAX_TOKENS` failures and no retries. Of 285
judgments, 254 were accepted with valid evidence and 31 were conservative
fallbacks; no invalid evidence reference was accepted.

Cold verification latency was 5,475.75 ms p50 and 8,752.25 ms p95, with an
11,226.62 ms maximum. A cache-only replay took 16.04 ms and made no provider
call. The final persistent ledger total is **$0.5026125**, including the older
unknown-usage failed-smoke reservation of $0.0498225. Remaining authorization
under $0.60 is **$0.0973875**. Complete local artifacts are
`results/relevance-smoke-vertex-grounded-v3-truncation-fix.json` and
`results/relevance-verifier-eval-v3.json`; the resumable checkpoint is retained.
No embeddings were regenerated, no discovery or alternate provider was used,
and no labels, deployment or production data were changed.

Final verification passed **168 tests**, including the isolated PostgreSQL/
pgvector integration test and all new truncation/usage-accounting regressions.
The only warning is the pre-existing Starlette `httpx` deprecation notice.
An initial integration launch used stale port 5433 and waited on connection
timeout; container inspection showed the healthy isolated database on 5432.
The run was stopped, the actual local endpoint was verified, and both the
integration test alone and the final all-in-one suite then passed.

## 2026-11-25 — Evidence references, bounded concurrency, and protected evaluation gate

### Evidence grounding

The relevance prompt/cache contract is now `relevance-verifier-v3-evidence-refs`
with cache format 2. Descriptions are deterministically split into exact,
numbered `E001`-style segments. The model receives untrusted segment text and
returns only batch-local candidate references plus evidence IDs. BoardForge,
not the model, resolves those IDs back to original text. Fabricated IDs,
duplicates, missing evidence, instruction-only evidence, and references without
substantive overlap with the role/intent/reasoning fail closed to `uncertain`.
The job content hash and prompt/cache versions invalidate resolved evidence when
source content changes. Deterministic eligibility restrictions remain outside
the model and cannot be upgraded.

Long stable job IDs were removed from model output in favor of `C001`-style
batch references. Reasoning is bounded to 160 characters. This reduces output
pressure without shortening evidence. Gemini retains a 6,000-character source
window, which is the same window used by the earlier successful 8/8-evidence
smoke; manually selected Groq retains its independent 12,000-character default.

### Latency and reliability

- Provider clients are reused: the Vertex SDK client was already lazy/cached,
  and the Groq HTTP client is now persistent.
- Independent verifier batches execute with `asyncio.gather`; provider
  semaphores continue to bound concurrency.
- Cache lookup occurs before planning or provider work.
- Verification metadata separately records eligibility, cache lookup, model
  wall time, summed provider inference, validation/reranking, and total time.
- Retrieval metadata separately records query embedding, database retrieval,
  ranking, and total retrieval latency.
- Gemini remains on documented `LOW` thinking. The default structured-output
  ceiling is now 768 tokens after the 640-token live response proved too small.
- Malformed Vertex responses now retain usage, latency, attempts, and cost when
  the SDK provides metadata, so paid malformed output is not reported as free
  in future runs.

### Central resource protection

The PostgreSQL/SQLite reservation ledger now includes conservative reserved
cost and a global cumulative cost ceiling. PostgreSQL uses one global advisory
lock so concurrent LLM, embedding, and discovery reservations cannot race.
Completed work replaces reservations with measured/estimated usage; in-flight
and unknown failed calls retain their reservations. LLM reservations include
the configured retry capacity and account for failed attempts conservatively.

Protection now covers BoardForge's production paths for:

- Vertex/Groq relevance generation (tokens, retries, in-flight calls and cost);
- Vertex embeddings in the API, worker, CLI indexing, comparison, real eval and
  holdout eval paths (each HTTP attempt is reserved); and
- Tavily discovery (per-minute, rolling-day request, and cumulative-cost gates).

The remaining limitation is that code outside BoardForge can instantiate a raw
provider without a store, and pricing is configuration-based rather than read
from a cloud billing ledger. Tavily's default cost is zero under the configured
free plan but its request-count ceiling is still enforced. Unknown failed calls
are intentionally over-counted until their rolling quota window expires.

### Resumable holdout gate and actual plan

The relevance evaluation now checkpoints atomically after each completed query.
It restores completed rows, reuses content/prompt-keyed judgments, reruns only a
partially completed query, and exits with `paused_budget_or_quota` rather than
waiting indefinitely. Candidate pools and frozen labels remain unchanged.

Using the current token-aware implementation on the frozen 100-job, 36-query
AI-audited dataset produced the actual plan:

- 285 uncached candidate judgments;
- 58 Gemini calls over the fixed per-query pools;
- 499,011 estimated input tokens;
- 30,880 reserved output tokens using candidate-count-aware caps up to 768;
- 7,424 maximum planned thinking tokens; and
- **$0.517898** estimated holdout inference cost.

This replaces the old extrapolation. No embeddings need regeneration.

### PostgreSQL verification and controlled smoke

A local pgvector PostgreSQL service was started and two isolated databases were
used. `boardforge_test` passed the complete integration test, including all
migrations through `007`, FTS/vector retrieval, evidence cache persistence and
atomic token/cost reservations. The full suite passed **164 tests with no
skips**; only the existing Starlette deprecation warning remains.

The one authorized optimized Gemini smoke used `boardforge_eval_v3`, the same
eight frozen `backend_general` candidates, `gemini-3.8-flash` in `global`, LOW
thinking, no provider retries, a 6,000-character evidence window, and a
640-token output ceiling. The request reached the provider but returned JSON
truncated in a reasoning string. Structured parsing therefore failed, all eight
candidates conservatively remained uncertain, and the holdout gate failed.
Observed verification wall time was 14,718.69 ms: eligibility 2.62 ms, cache
20.79 ms, model phase 14,672.54 ms, validation/reranking 0.25 ms. The older SDK
path raised during parsing before exposing response metadata, so reported
tokens/cost were zero; the persistent ledger conservatively retained 13,286
reserved tokens and **$0.0498225** instead of assuming the call was free.

After the failed smoke, output reservations were made candidate-count-aware;
the failed reservation plus the final $0.517898 plan totals **$0.5677205**,
inside the monetary ceiling. The complete holdout was nevertheless not started
because structured output and evidence grounding did not pass the mandatory
smoke quality gate. No Gemini retry, Groq fallback, embedding call, Tavily call,
discovery, label change, deployment, or billing change occurred. The guarded
smoke artifact is preserved locally at
`results/relevance-smoke-vertex-grounded-v3.json`; changes remain uncommitted.

## 2026-11-24 — Groq token-aware batching, persistent limits, and successful retry

### Problem and design

The earlier eight-candidate Groq request failed with HTTP 413 because one fixed
batch was larger than the account's published 8K tokens-per-minute allowance.
Reducing every description to 3,000 characters would discard useful evidence,
so that temporary mitigation was replaced rather than retained.

The verifier now estimates the exact serialized Groq request, including the
strict JSON schema, adds a conservative 8% estimation margin, reserves the
bounded output allowance, and greedily forms batches of at most four. It splits
the batch before shortening descriptions; only a single candidate that cannot
fit by itself may be truncated. The production defaults preserve up to 12,000
description characters, cap a request reservation at 7,600 tokens, and keep
the model fixed at `openai/gpt-oss-120b`. There is no model routing or Gemini
fallback.

SQLite and PostgreSQL now share a persistent `llm_usage_reservations` ledger.
Reservations are atomic per provider/model and enforce 8,000 TPM and 200,000
tokens per rolling 24 hours before the network call. A completed call replaces
its conservative reservation with actual provider usage; a failed call retains
the reservation because provider-side consumption is unknown. PostgreSQL uses an advisory transaction lock, while
SQLite uses `BEGIN IMMEDIATE`. Per-search call/token limits remain a second
guard. Normal searches fail closed when quota is unavailable; the controlled
smoke explicitly allowed up to 70 seconds of waiting for a minute window.

### Controlled retry

The same frozen `backend_general` candidate IDs used by the Gemini smoke were
retried once through Groq. The no-network plan produced 3/3/2 candidate batches
with 12,000-character evidence windows. Provider retries were zero.

- logical calls / HTTP attempts / retries: **3 / 3 / 0**;
- HTTP 413 or other provider failures: **none**;
- structured-output compliance: **yes for all three responses**;
- effective validated judgments: **4 direct, 1 irrelevant, 3 uncertain**;
- exact evidence validation: **5/8 passed**; the other three were conservatively
  changed to uncertain because their returned snippets were not exact source
  substrings;
- input / answer / thinking tokens: **10,007 / 898 / 1,082**;
- total tokens: **11,987**;
- provider latencies: **1,834.53 / 2,230.42 / 2,031.90 ms**
  (**6,096.85 ms** generation total);
- quota pacing: calls began about 60 seconds apart; end-to-end wall time was
  about **123.18 seconds**, mostly deliberate TPM waiting;
- estimated inference cost: **$0.00268905** at $0.15/M input and $0.60/M
  completion tokens, including reasoning tokens.

The validated direct results were Perplexity Search Crawler, Ramp Ops
Engineering, Perplexity Search Backend Infra internship, and Cloudflare Fintech.
Roblox's engine/graphics role was correctly excluded as irrelevant. Both
Palantir backend postings and the Roblox early-career posting failed exact
evidence validation and remained `needs_verification`; this demonstrates the
intended conservative boundary rather than silently trusting near-match text.

Gemini's earlier identical-candidate smoke returned 6 direct and 2 adjacent
with 8/8 evidence validation, used 9,653 tokens, took 11.72 seconds of provider
latency, and cost $0.01068375. Groq was about 3.97x cheaper and had lower pure
generation latency, but its exact-evidence reliability was worse and free-tier
pacing made end-to-end latency much longer.

At the measured per-call average, a 58-call evaluation is estimated at about
193,469 input, 17,361 answer, and 20,919 thinking tokens (231,749 total) and
**$0.05199**. This exceeds the configured 200K-token rolling-day limit, so it
must not run in one day under the current free-tier guard. The full holdout was
not run.

Artifacts are locally ignored at
`results/relevance-smoke-groq-token-aware-v2.json` and its SQLite cache/ledger.
The credential was only loaded into process memory and was never printed.
Final verification: **161 passed, 1 skipped** (external PostgreSQL integration
database unavailable), with the existing Starlette warning. No embeddings,
discovery, billing, deployment, label changes, or holdout evaluation occurred.

## 2026-11-24 — Groq alternative-provider refresh and smoke diagnosis

The locally configured relevance key was detected without reading it into logs
or printing its value. Before making a request, current Groq documentation was
checked. `llama-3.3-70b-versatile`, the old default, was shut down for free and
developer accounts on 2026-08-16. The adapter now defaults to its recommended
replacement, `openai/gpt-oss-120b`, which supports strict JSON schemas and is
listed at $0.15/M input and $0.60/M output tokens.

Strict-mode schema normalization was added because Groq requires every field to
be required and every object to set `additionalProperties: false`. A mocked test
initially exposed the missing object restriction; normalization was corrected
before any live call, and 23 focused tests passed.

One guarded eight-candidate smoke request was then attempted with retries forced
to zero. It returned HTTP 413 `Payload Too Large` before generation:

- logical calls / HTTP attempts / retries: **1 / 1 / 0**;
- reported input / output / thinking tokens: **0 / 0 / 0**;
- estimated inference cost: **$0.00**;
- all eight rows conservatively became `needs_verification`; and
- no second live request was made.

Local request measurement found the 6,000-character payload was 41,196 JSON
bytes, approximately 10,299 tokens. Groq's published base limit for
`openai/gpt-oss-120b` is 8K TPM, making request size/quota the most likely cause
of the 413, although the response did not include a more specific subcode. The alternative-provider
defaults were therefore reduced to 3,000 description characters per candidate
and 1,280 output tokens: the same batch is approximately 6,436 input tokens and
under 8K including its bounded output. This fix is covered locally but remains
unverified by a second live call, deliberately respecting the one-call control.

The guarded failure artifact is stored locally in ignored
`results/relevance-smoke-groq.json`. No credential value, holdout evaluation,
embedding operation, discovery, or deployment occurred. Final validation after
the fallback changes: **160 tests passed** with the existing Starlette warning;
Docker Compose, deployment-script parsing, and `git diff --check` passed.

## 2026-11-24 — Vertex Gemini relevance provider and controlled smoke test

### Provider selection and availability

The primary relevance provider now uses the official `google-genai` SDK in
Vertex AI mode with ADC. A read-only availability check was made against project
`boardforge-ai-2026`. The SDK's generic `models.list()` path returned a 404
because Vertex publisher models do not expose that collection endpoint. No
generation occurred. Model IDs were then checked with the non-generative
`countTokens` endpoint:

- available in `global`: `gemini-3.8-flash`, `gemini-3.7-flash`, and
  `gemini-3-flash-preview`;
- unavailable in `global`: `gemini-3.6-flash` and `gemini-2.5-flash`; and
- the checked Flash IDs were unavailable in `us-central1`.

`gemini-3.8-flash` in `global` was selected as the stable, capable Flash model.
The model and location remain configurable. The current documented introductory
pricing used for estimates is $0.75 per million input tokens and $3.75 per
million output/reasoning tokens through 2026-12-31.

### Implementation and controls

- Added a reusable provider-neutral generation request/result abstraction with
  structured schemas, token usage, tool declaration support for future agent
  reasoning, latency, attempts, and cost metadata.
- Added `VertexGeminiLLMProvider` using native JSON response schemas, ADC, the
  stable Vertex `v1` API, bounded explicit retries, timeout, concurrency,
  maximum output tokens, and `LOW` thinking level for Gemini 3.8. Older models
  that support integer control retain a bounded thinking-budget fallback. SDK-internal retries are
  disabled so BoardForge counts attempts and never retries a paid malformed
  response.
- Kept the OpenAI-compatible/Groq relevance adapter as an alternative. Vertex
  is now the default and needs no Gemini API key.
- Search-level limits default to three calls and 60,000 accounted tokens. Rows
  beyond a limit fail closed as `needs_verification`; ingestion and workers do
  not invoke the model.
- Relevance metadata now records input, output, thinking and cached tokens,
  provider latency, attempts/retries, estimated cost, structured judgments, and
  evidence-validation status. Existing cache keys continue to include job
  content, intent, preferences, model, and prompt version.
- Cloud Run configuration explicitly enables the verifier only with
  `-EnableRelevanceVerifier`. Vertex uses the runtime service account's existing
  `roles/aiplatform.user`; only the optional Groq path mounts an API-key secret.

### Exactly-one live smoke batch

After mocked and full tests passed, one guarded batch was run from the frozen
real corpus for intent `backend software engineer`. Retries were forced to zero.
The guard created dedicated cache/result artifacts and refuses to run again if
either exists.

- logical calls / HTTP attempts / retries: **1 / 1 / 0**;
- candidates: **8**;
- structured-output compliance: **yes**, all eight judgments parsed against the
  native schema;
- evidence validation: **8/8 passed** exact substring validation;
- judgments: **6 direct**, **2 adjacent**, **0 irrelevant**, **0 uncertain**;
- input / output / thinking tokens: **8,505 / 1,148 / 0**;
- total tokens: **9,653**;
- provider latency: **11,715.06 ms**;
- estimated request cost: **$0.01068375**; and
- provider/API failures: **none**.

The two adjacent results were a specialized Roblox game-engine internship and a
general early-career software-engineering posting; both remained alternatives
rather than direct backend recommendations. Complete machine-readable output is
in the ignored local `results/relevance-smoke-vertex.json` artifact. Labels
remain AI-audited, not human-verified.

The live request accepted the older integer thinking-budget and temperature
fields, but a final official-model-guide check identified them as deprecated for
Gemini 3.8. The implementation was subsequently aligned to
`thinking_level=LOW` with temperature omitted and covered by a mocked regression
test. No second live request was made.

### Full-holdout cost estimate and verification

The smoke batch used 87.0% of its character-based planned input estimate and
119.6% of its planned output estimate. Applying those measured ratios to the
frozen 58-call plan gives approximately 337,942 input and 40,898 output tokens,
zero projected thinking tokens if behavior remains similar, and **$0.4068** in
inference charges. A cautious allowance for mix/length variance and 128
thinking tokens per call as a cost-planning reserve puts the ceiling around **$0.45–$0.50**,
well below the remaining project budget. This is an extrapolation from one
batch, not a billing guarantee.

Final validation with the isolated PostgreSQL test URL: **158 passed**, one
pre-existing Starlette deprecation warning. Docker Compose validation,
PowerShell deployment-script parsing, and `git diff --check` are also required
in the final handoff. No holdout evaluation, embedding regeneration, discovery,
deployment, or label modification was performed.

## 2026-11-24 — Bidirectional relevance-verifier revision

### Why the first boundary was insufficient

The first verifier saw only candidates already labeled `recommended`. That
could reduce false positives but could never recover a relevant candidate that
the deterministic confidence policy conservatively marked
`needs_verification`. It also ran after top-K truncation, so a corrected result
could not move into the final selection.

### Revised design

- Retrieval supplies a bounded pre-verification pool when the feature is
  enabled. Verification runs before the API applies the requested final top-K.
- The shortlist includes deterministic recommendations and plausible
  `needs_verification` candidates. Production defaults to at most 20 candidates
  in batches of eight; the frozen saved holdout pools contain at most ten.
- Eligibility is separate from relevance. Rejected-job memory, hard violations,
  unknown required salary/experience eligibility, and descriptions explicitly
  confirming no current vacancy are never sent for an upgrade.
- Supported `direct` becomes a recommendation; `adjacent` has a distinct
  `alternative` status; `irrelevant` is omitted; `uncertain` and conservative
  failures remain `needs_verification`.
- Final ordering is direct, adjacent, then uncertain, preserving score order
  within each group. Relevant candidates can therefore move up before top-K.
- Prompt version `relevance-verifier-v2` invalidates v1 cache entries. Candidate
  descriptions are explicitly untrusted JSON data. Instructions inside them
  cannot change the verifier rules, and evidence must occur in the truncated
  description actually supplied to the provider.
- Evaluation starts from identical saved hybrid pools and now reports
  precision/recall deltas, recoverable relevant uncertain candidates, recovered
  recommendations, and recovery rate.

### Planned provider use (no calls executed)

The frozen 100-job, 36-query, v1.1 AI-audited holdout produces this empty-cache
plan from its existing saved candidate pools:

- 285 candidate judgments;
- 58 model calls, with at most eight candidates per call;
- approximately 388,445 input and 34,200 output tokens (422,645 total);
- about 87 seconds sequentially at the 1.5-second planning assumption; and
- about 3 seconds added at p50 and p95 for the saved pools when uncached.

These are estimates, not measurements. No paid relevance or Vertex request was
made. Execution still requires `relevance-eval --execute`, an isolated
PostgreSQL URL, provider configuration/key, and explicit authorization.

### Deployment, verification, and issue encountered

The feature remains off by default. Docker uses the explicit environment flag.
The GCP deployment script requires `-EnableRelevanceVerifier`; only then does it
mount the relevance secret and enable verification for the API service.

Regression coverage includes upgrades, eligibility restrictions, adjacent
alternatives, malicious description instructions, exact-evidence failure,
missing/malformed judgments, provider outage, bounded batches, reordering,
cache behavior, and recovery metrics. The full suite passed: **152 tests**, with
one pre-existing Starlette warning. `git diff --check` and Docker Compose
configuration validation also passed.

The session's declared workspace `E:\mnt\data\matcheval` did not exist; the real
checkout was `E:\boardforge`. Pytest initially failed during fixture setup
because it could not create temporary files. Re-running with a base-temp under
the actual checkout and approved filesystem access resolved this environment
problem; it was not a product-code failure.

## 2026-11-24 — Initial LLM relevance-verification layer (superseded)

### Motivation and boundary

The controlled holdout showed strong hybrid candidate recall (0.9437) but weak
recommendation confidence: 0.4789 recommended precision and 0.5211 recommended
irrelevant rate, plus eight false recommendations on four no-relevant-result
queries. Deterministic hard-constraint violations remained zero.

This section records the original implementation before the bidirectional
revision above. That layer preserved retrieval and deterministic confidence. It ran only
after ranking and is downgrade-only: the LLM can verify a deterministic
recommendation as direct/adjacent or demote it to `needs_verification`. It can
never upgrade an existing uncertain candidate, restore rejected feedback, waive
a hard constraint, or resolve unknown salary/experience eligibility.

### Implementation

- Added a provider-independent structured batch protocol with an included
  OpenAI-compatible/Groq adapter and dependency-injected mock support.
- The prompt asks for direct, adjacent, irrelevant, or uncertain plus concise
  reasoning and verbatim description snippets. It uses general annotation
  principles, not individual holdout failures.
- Every supplied evidence snippet is checked as an exact source-description
  substring. Missing, hallucinated, duplicate, omitted, or malformed judgments
  become uncertain and are not cached.
- Company boilerplate, optional qualifications, incidental technologies,
  occupational mismatch, and explicit no-vacancy postings are called out as
  generic decision boundaries.
- Provider errors/timeouts fail closed for the shortlisted recommendations:
  they become `needs_verification`. When the feature flag is off, existing
  deterministic behavior is unchanged.
- Verification is bounded to ten deterministic recommendations by default and
  batches up to eight candidates per call.
- Durable SQLite and PostgreSQL caches key judgments by job-content hash,
  normalized intent, canonical preferences, model, and prompt version. A new
  PostgreSQL migration creates `relevance_verifications`.
- `JobAgent.search_jobs` and resume recommendations attach verification
  metadata to the retrieval trace. Candidate scores and retrieval sets are
  preserved; ordering is stable within verified/needs-verification groups.

### Identical-candidate evaluation mode

`boardforge relevance-eval` defaults to a no-call plan. A future explicitly
authorized execution reconstructs the already saved hybrid top-10 candidate
sets, applies verification, asserts set equality, and compares deterministic
and verified recommendation metrics. It does not run discovery or retrieval
again.

For the current 36-query AI-audited holdout and an empty judgment cache:

- candidate judgments: 150;
- proposed model calls: 31;
- batch size: at most 8;
- estimated input tokens: 206,824;
- estimated output tokens: 18,000;
- estimated total: 224,824 tokens;
- estimated sequential evaluation latency: 46.5 seconds; and
- estimated uncached production addition: 1.5 seconds p50 / 3.0 seconds p95,
  assuming 1.5 seconds per provider call.

These are planning estimates, not measured provider results. No relevance-model
API call was made.

### Verification and limitations

- Final suite: 146 passed, including PostgreSQL integration and mocked valid,
  irrelevant, unsupported-evidence, hallucination, missing-candidate,
  malformed-output, timeout, bounded-batch, cache, and deterministic-protection
  cases. One pre-existing Starlette deprecation warning remains.
- The labels are AI-audited, not human-verified. A model judge can reproduce
  annotation biases even when its evidence quote is real.
- Exact snippet validation proves provenance, not that the reasoning drawn from
  the snippet is correct.
- Description truncation bounds tokens but can omit decisive evidence later in
  a long posting. The conservative outcome is uncertainty.
- Downgrade-only verification can improve precision only by reducing coverage;
  it cannot recover retrieval misses or promote relevant uncertain results.
- Latency and structured-output compatibility vary by provider/model. Cache hit
  rate is therefore operationally important.

## 2026-11-24 — Controlled v1.1 holdout evaluation

### Controls and execution

The frozen 100-job corpus and `v1.1-ai-audited` labels were checksum-validated
before execution. The labels remain explicitly **AI-audited, not
human-verified**. Ranking code and labels were not changed before or after
inspection.

One authorized Vertex run populated a shared task-aware cache with 100 document
and 36 query embeddings. All 136 HTTP attempts returned 200; there were zero
retry requests. Lexical, hybrid, and exhaustive variants made zero incremental
paid requests. No discovery, deployment, second experiment, or embedding
backfill outside the isolated holdout was run.

Cold timing was 27,827.86 ms for document embeddings and 12,683.60 ms for query
embeddings (40,511.46 ms total provider time), followed by 5,733.47 ms to write
document vectors to PostgreSQL. Warm search timings exclude embedding creation,
metric-only candidate queries, and `EXPLAIN ANALYZE`.

### Queries with relevant jobs (32)

| Metric | Lexical | Hybrid PostgreSQL | Exhaustive |
|---|---:|---:|---:|
| Candidate recall | 0.6681 | 0.9437 | 1.0000 |
| Precision@10 | 0.2406 | 0.3312 | 0.3250 |
| Recall@10 | 0.3791 | 0.5607 | 0.5528 |
| NDCG@10 | 0.4210 | 0.5771 | 0.5720 |
| Graded NDCG@10 | 0.4192 | 0.5857 | 0.5825 |
| Recommended precision | 0.4087 | 0.4789 | 0.4626 |
| Recommended recall | 0.2035 | 0.2944 | 0.2944 |
| Recommendation coverage | 0.5044 | 0.5591 | 0.5787 |
| Abstention rate | 0.4956 | 0.4409 | 0.4213 |
| Recommended irrelevant rate | 0.5913 | 0.5211 | 0.5374 |
| Forbidden-result rate | 0.0219 | 0.0236 | 0.0236 |
| Hard-constraint violation rate | 0.0000 | 0.0000 | 0.0000 |
| Warm latency p50 / p95 (ms) | 73.09 / 118.60 | 143.75 / 328.16 | 150.85 / 546.74 |

Candidate recall and ranking metrics are macro averages. Recommendation and
constraint rates are micro averages over returned/recommended results.

### Queries with no relevant jobs (4)

The no-result intents are technical recruiter, legal counsel, COBOL mainframe,
and biomedical wet lab. They are excluded from recall and NDCG aggregation.
COBOL and biomedical correctly produced no confident recommendations in all
variants. Technical recruiter and legal counsel did not:

- lexical: 9 false recommendations across two queries;
- hybrid: 8 false recommendations across two queries; and
- exhaustive: 11 false recommendations across two queries.

Every recommendation in this group is necessarily irrelevant, so recommended
irrelevant rate is 1.0 for all variants. Hybrid reduced false recommendations
versus lexical but did not solve abstention on empty ground truth.

### Retrieval, confidence, and plan findings

- Hybrid improved candidate recall by 0.2756 absolute over lexical and closely
  matched the exhaustive reference on final ranking metrics. The remaining
  0.0563 candidate-recall gap caused only a small final difference here.
- Confidence remains the principal weakness: hybrid made 74 false confident
  recommendations among 142 recommendations on relevant queries (52.11%
  irrelevant), while preserving zero deterministic hard-constraint violations.
- The largest false-recommendation clusters were product designer, technical
  support, remote support, security engineer, site reliability, streaming
  pipelines, data-plumbing paraphrase, and SDET frameworks.
- `EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)` was captured for all 36 lexical and
  hybrid queries. PostgreSQL chose neither GIN nor HNSW for any query at this
  100-row scale. Lexical plans used no named index; hybrid plans used primary-key
  indexes on some queries. Raw executed plans are preserved in the JSON report;
  no index-usage claim is inferred from index existence.
- Seven label decisions have unresolved eligibility. Three junior-experience
  candidates were retrieved but never recommended; the other unresolved
  country, work-policy, and experience rows were neither returned nor
  recommended. These are reported separately rather than scored as confirmed
  eligibility failures.

### Reporting correction and artifacts

The first summary macro-averaged recommendation rates with zero-recommendation
queries, which made the no-relevant recommended-irrelevant rate appear as 0.5.
Per-query outputs were correct. The aggregate layer was corrected to use micro
counts and the saved rows were deterministically reprocessed. Retrieval and
Vertex were not rerun.

- Machine-readable result: `results/holdout-eval-v1_1-ai-audited.json`
- Readable comparison: `results/holdout-eval-v1_1-ai-audited.md`
- Embedding cache: `results/holdout-embedding-cache.json`

Final verification: 136 tests passed, including PostgreSQL/pgvector
integration. One pre-existing Starlette deprecation warning remains.

## 2026-11-24 — Independent semantic audit of holdout labels

### Trigger and scope

Independent review found that provisional regex evidence had contaminated
task-specific labels. Examples included CI, AV automation, mobile-product
management, and general software engineering being treated as vision or LLM
inference work because their descriptions mentioned company research or an
optional technology.

The original corpus and provisional label file were preserved byte-for-byte.
A separately versioned `1.1-ai-audited` dataset was produced. This is explicitly
an **AI audit, not human verification**.

The pass reviewed:

- all 511 provisional direct, adjacent, or forbidden decisions;
- 176 provisional-irrelevant decisions sampled across every query and spread
  across companies (five per query where the available strata allowed it);
- 38 additional provisional-irrelevant decisions surfaced as plausible missed
  positives during the responsibility inventory;
- every posting for explicit no-vacancy language; and
- the full relevant candidate set for country, remote-policy, experience, and
  negated-role constraints.

No ranking prediction, ranking output, `confidence.py`, regex relevance result,
Vertex call, job collection, or ranking-code change was used in the audit.

### Annotation policy

- Direct relevance requires the requested role/task to be a substantive duty.
- Adjacent relevance requires a neighboring occupational responsibility that
  would plausibly help the same user, not merely shared terminology.
- Company descriptions, research portfolios, optional qualifications, and
  one-off technology mentions cannot establish relevance.
- Management and individual-contributor roles are separate occupational
  intents. A manager is not automatically adjacent to every engineer on the
  managed technology.
- A constraint is forbidden only when explicit and confirmed. Unknown country,
  work policy, or numeric experience is retained with `eligibility=unresolved`,
  not converted into ineligibility.
- “No management” and “instead of backend” are explicit negations. Broad role
  wording without negation is a preference, not a hard prohibition.
- Anthropic's Expression of Interest posting states that there is no open
  Research Manager position. It is forbidden across queries even though the
  description discusses real interpretability research.

### Results

Across all 3,600 label cells, the audit made 396 corrections and agreed on
3,204 (89.0% overall agreement). The audited distribution is 64 direct, 167
adjacent, 3,299 irrelevant, and 70 forbidden, with seven unresolved eligibility
decisions.

Within the specifically requested 511 non-irrelevant decisions, 324 changed
(63.4% disagreement) and 187 remained unchanged. In the 176-row stratified
irrelevant sample, seven changed, including two missed positives. The broader
responsibility inventory found 38 missed positives in total.

Important transitions:

- 59 adjacent → irrelevant and 18 direct → irrelevant, primarily boilerplate,
  optional qualifications, incidental tools, or occupational mismatch;
- 205 forbidden → irrelevant, primarily implicit preferences or unrelated roles
  that had been incorrectly treated as hard-constraint violations;
- 30 irrelevant → adjacent and 8 irrelevant → direct, representing substantive
  responsibilities missed by the provisional rules;
- 34 irrelevant → forbidden, mostly the explicit no-current-vacancy posting and
  confirmed constraints.

The four reported contamination examples are now irrelevant. Queries with no
relevant jobs are technical recruiter, commercial legal counsel, COBOL
mainframe programmer, and biomedical wet-lab technician. Additional queries
with no direct result include LLM inference optimization, ML platform,
computer-vision perception, streaming pipelines, SDET frameworks, site
reliability, and the intentionally ambiguous technical-role request.

### Artifacts and remaining work

- Dataset: `boardforge/data/holdout_labels_v1_1_ai_audited.json`
- Correction/evidence report: `docs/holdout-v1.1-ai-audit.md`
- Audited checksum:
  `c8cfc4ea4043bf9db48820c26efac3d63488231ca6664f3afd7688954b9b843c`

The seven unresolved cases remain intentionally unresolved until a human checks
the source posting or company eligibility policy. This version must not be
described as human-verified.

Verification: the final full suite, including the dedicated PostgreSQL/pgvector
integration test, passed 135 tests. One pre-existing Starlette deprecation
warning remains. No Vertex or discovery call was made.

## 2026-11-24 — Independent holdout and production-retrieval evaluation

### Requested outcome

Create an independent real-job holdout, provisional evidence-backed labels, and
an evaluation path through the production PostgreSQL retrieval code. Stop
before the paid Vertex run, preserve India v1.1 unchanged, and keep all work
uncommitted for review.

### Dataset decisions

- The source manifest contains ten targeted official ATS boards: Anthropic,
  Cloudflare, Datadog, Figma, Roblox, Palantir, Notion, Linear, Ramp, and
  Perplexity. It uses only existing Greenhouse, Lever, and Ashby connectors and
  excludes the development companies Coinbase, Jumio, and Twilio.
- Collection is deterministic: each company receives an equal quota and jobs
  are selected by a category round-robin with stable title/location/ID ordering.
  Collection refuses to overwrite an existing snapshot unless `--force` is
  explicit.
- The frozen corpus contains 100 jobs from ten companies. It stores full job
  descriptions, official application and board URLs, stable IDs, collection
  timestamps, category/seniority metadata, and a canonical SHA-256 checksum.
- The 36 intents cover broad and specialized roles, task paraphrases, changing
  preferences, manager-versus-IC intent, hard geography/work-policy/experience
  constraints, ambiguous requests, nontechnical distractors, and two intended
  no-suitable-job cases.
- Labels are generated from explicit per-intent description/title criteria and
  explicitly declared structured hard constraints. They were generated before
  any ranking evaluation and import neither production query parsing nor the
  confidence policy. Each of the 3,600 decisions is one of
  direct, adjacent, irrelevant, or forbidden.
- Labels are deliberately marked `automated_provisional_not_human_verified`.
  The Markdown review export includes intent notes and evidence snippets for
  every non-irrelevant decision. Human corrections must create a new version;
  they must not mutate the frozen provisional file.

### Evaluation infrastructure

- Added a guarded evaluation runner for lexical-only production retrieval,
  hybrid PostgreSQL retrieval, and exhaustive corpus reranking.
- The runner refuses database names that do not contain `holdout`, `eval`, or
  `test`, and refuses an evaluation database containing unrelated active jobs.
- Document and query embeddings share a persistent, task-aware cache keyed by
  model, task type, and text hash. Provider batch calls, provider prediction
  requests, and cache hits are recorded.
- Metrics include candidate recall, final precision/recall/NDCG,
  recommendation precision/recall/coverage, abstention, forbidden-result rate,
  hard-constraint violations, p50/p95 latency, and embedding usage.
- `PostgresStore.explain_candidate_jobs` executes the exact candidate SQL under
  `EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)` and reports only indexes named by
  executed plan nodes. It does not infer usage from index existence.

### Issues encountered and resolution

- The environment supplied `E:\mnt\data\matcheval`, but that path did not exist;
  the live repository was `E:\boardforge`. Work continued on the verified
  `feat/boardforge-v2` checkout.
- System Python lacked project dependencies. All generation and tests were
  rerun with the repository `.venv`.
- The first category classifier searched entire descriptions, so corporate
  boilerplate made sales roles look like AI jobs. The classifier was corrected
  to prioritize role-bearing title evidence and use a restricted technical-only
  description fallback. The corpus was recollected once before labels were
  created or evaluation was run.
- Managed filesystem permissions initially blocked generated labels and the
  test SQLite database. The same commands succeeded with scoped repository
  write permission; application code was not changed to work around the
  environment.
- A real local `boardforge_holdout_eval` database was created separately from
  the application database. With the 100 frozen rows and synthetic (non-paid)
  vectors, PostgreSQL used `idx_jobs_status` and `job_embeddings_pkey`; it did
  **not** choose the GIN or HNSW indexes. This is the honest result for this
  small corpus/current query shape, not an index-usage claim. The full runner
  preserves the raw plans so the paid evaluation can recheck the actual plans.

### Verification and stop condition

- Frozen corpus checksum:
  `d1f7a5bdbf65dceea31b6c5f67143a4c633c20fdaff88b5748b4837524e139e3`.
- Provisional-label checksum:
  `95166916abf9fcc1d221df3f94e06bde8c4a65dd952b98872305b4d1e31bb020`.
- Final full suite with a dedicated `boardforge_test` pgvector database: 130
  passed with one existing Starlette deprecation warning. The production
  EXPLAIN path was additionally exercised manually against the isolated local
  holdout database.
- Planned paid usage with an empty cache: 100 document + 36 query = **136
  Vertex prediction requests**. No Vertex evaluation was executed, in
  accordance with the requested stop point.
- Remaining human action: review `docs/holdout-v1-label-review.md`, record
  corrections in a new human-verified label version, approve the 136-request
  Vertex run, and then interpret retrieval/ranking metrics without tuning
  `confidence.py` against this holdout.

## 2026-11-24 — Deterministic ranking reliability milestone

### Requested outcome

Reduce false confident recommendations without damaging retrieval recall. The
starting frozen India evaluation contained 29 jobs and 24 queries (dataset
v1.1): Top-1 1.0, Recall@10 0.9315, NDCG@10 0.9406, recommended-irrelevant rate
0.3042, and hard-constraint violation rate 0.0.

### Problems found

1. Confidence enforcement lived in `hybrid_rank_jobs`, after the query embedding
   succeeded. Lexical-only searches, missing document vectors, disabled
   embeddings, and provider outages therefore used weaker confidence rules.
2. Query evidence used a set intersection over the entire description. Words
   from unrelated paragraphs could combine into apparently strong evidence.
3. The query embedding was requested in `retrieve_and_rank_jobs`; after failure,
   `hybrid_rank_jobs` attempted the same request again.
4. Role-only filtering removed potentially useful uncertain candidates instead
   of retaining them with an explicit verification status.
5. Evaluation measured false recommendations but did not expose the precision,
   recall, coverage, or abstention trade-off of the confidence policy.

### Design decisions

- Confidence is a deterministic policy over structured job/query evidence. An
  embedding may change ordering, but it may not upgrade unsupported relevance.
- Query intent is separated into role, domain, skill, and task evidence.
- Description evidence must co-occur inside a short token window. Scattered
  terms across a long JD count as retrieval hints, not verification.
- Broad words such as “platform”, “data”, and “services” cannot establish
  specific intent when the distinctive query terms are absent.
- Leadership and individual-contributor titles are checked independently of
  semantic similarity. “Engineering manager” and “software engineer” are not
  treated as interchangeable.
- Uncertain candidates remain in the result set as `needs_verification`. This
  separates retrieval recall from recommendation precision instead of hiding
  ambiguity by deleting results.
- No company, job ID, or evaluation-case-specific rule is used. The policy uses
  reusable occupational categories and evidence geometry.

### Implementation

- Added `boardforge/confidence.py` as the shared policy and intent assessment
  module.
- Applied the policy inside deterministic `rank_jobs`, before any semantic
  branch can return.
- Removed the semantic-only confidence implementation.
- Added a `query_embedding_attempted` signal so a failed initial request cannot
  be retried accidentally during hybrid rescoring.
- Added categorical `query_domain`, `query_skill`, and `query_task` evidence.
- Added recommended precision/recall, recommendation coverage, abstention rate,
  and no-confident-recommendation count/rate to the real evaluation report.

### Issues encountered and resolution

- Existing tests expected role-mismatched jobs to disappear. The milestone
  requires preserving *relevant* uncertain candidates, not retaining candidates
  with no evidence for an explicitly requested role. An initial implementation
  kept all role mismatches and degraded exact role searches and benchmark
  precision. The explicit role gate was restored; uncertainty is preserved
  inside the plausible role/domain candidate set.
- A salary-confidence test had one candidate with verified salary but no query
  evidence and another with query evidence but unknown salary. Both are now
  correctly uncertain for different reasons; the test checks preservation and
  status rather than forcing a misleading winner.
- A dedicated real-eval fixture was added where a labeled relevant candidate is
  retained as uncertain. It verifies zero recommendation coverage, full
  abstention, and one query with no confident recommendation without confusing
  that behavior with a clear explicit-role mismatch.
- The first evaluation command passed the corpus as a positional argument, but
  the CLI requires `--corpus`. Argument parsing stopped before provider setup,
  so no paid request occurred. The corrected command was then run once.

### Verification

- Focused confidence and evaluation regressions: 38 passed during development.
- Final full suite: 124 passed; one existing Starlette `TestClient` deprecation
  warning remains.
- One Vertex run was made over the frozen 29-job/24-query corpus. No discovery
  or bulk index backfill was run.

| Metric | Before | After |
|---|---:|---:|
| Top-1 accuracy | 1.0000 | 1.0000 |
| Top-1 direct accuracy | 0.9583 | 1.0000 |
| Recall@10 | 0.9315 | 0.9315 |
| NDCG@10 | 0.9406 | 0.9366 |
| Graded NDCG@10 | 0.9375 | 0.9408 |
| Recommended irrelevant rate | 0.3042 | 0.1542 |
| Hard-constraint violation rate | 0.0000 | 0.0000 |

New confidence metrics: recommended precision 0.8458, recommended recall
0.7755, recommendation coverage 0.4412, abstention rate 0.5588, and 0/24
queries with no confident recommendation.

The recommended-irrelevant rate fell by 49.3% relative while retrieval recall
and Top-1 accuracy were preserved. Binary NDCG decreased by 0.0040, while graded
NDCG and direct Top-1 accuracy improved.

### Remaining measured failure clusters

- Distinctive domain terms can still be outweighed by broad words such as
  “platform”, “services”, and “operations” in security and data queries.
- Some title families remain too close: account management versus support
  engineering, task-focused IC work versus management, and enterprise
  architecture versus implementation.
- Eight queries still contained at least one irrelevant confident
  recommendation: biometrics data pipelines, data engineer/Airflow, Segment
  data pipelines, storage/streaming platform, security platform backend,
  security incident response, programmable messaging support, and technical
  account management.
- Four additional failure rows contained evaluation-forbidden retrieved
  candidates but no false confident recommendation. These are retrieval/ranking
  issues, not hard-constraint violations.

A stricter distinctive-domain/title-family pass was prototyped locally after
the paid run, but deliberately not retained: changing status can change ranking,
so shipping it without another frozen Vertex evaluation would make the reported
metrics inaccurate. It is recorded here as the next measured iteration rather
than presented as an unverified improvement.

### Remaining external work

- A real Vertex evaluation requires Application Default Credentials, the GCP
  project, and paid API access. It should be run once after local tests pass to
  avoid repeated paid calls.
