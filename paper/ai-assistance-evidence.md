# AI-assistance evidence for author disclosure

AI assistants supported programming, debugging, experiment coordination, documentation and manuscript preparation. The author remains responsible for the scientific claims, implementation, citations and submitted text. This note distinguishes recorded execution from model names mentioned in requests; it is an evidence inventory, not a claim that every generated contribution was correct or retained.

| Model name | Evidence status | Scope and limitation |
|---|---|---|
| GLM-5.3-Flash | Recorded executions | The project-specific ZCode model-usage metadata records5,830 completed requests,36 cancelled and19 errors under this exact model identifier. These are model requests, not independent experiments or accepted code changes. |
| GLM-5.3 | Recorded executions | The same metadata records3 completed requests under this exact identifier. Its use is therefore supported independently of user requests. |
| DeepSeek Flash V4 | Author clarification and recorded launcher configuration | The author specified DeepSeek Flash V4 on September 21. Launcher receipts identify `deepseek-flash` and sometimes `deepseek-official/deepseek-flash`; these identify the configured backend but do not independently establish a finer version. |
| GPT-6 Astra | Reported assistance; exact runtime attribution incomplete in this audit | Historical operational messages attribute reviews and a separate stricter campaign to an assistant named Astra. That name alone does not prove an exact model revision. The ZCode usage table contains only the GLM identifiers above; separate orchestration evidence is needed to bind the reported GPT-6 Astra model name to individual contributions. |

Stored backend identifiers do not independently attest a provider's internal weights or version mapping. Requesting a model, naming a model in a handoff, and recording a completed provider request are different kinds of evidence. The manuscript should use the level of specificity supported by the records and describe substantive assistance, rather than presenting a model roster as verification of the research.

No private reasoning, credentials, or raw conversation excerpts are included here. Detailed accounting remains in the private provenance archive. The recorded request counts cover the historical ZCode project sessions inspected, not the entire multi-tool campaign.

## Final analysis pass, 2026-09-21

The Pi harness invoked provider `opencode-go`, model identifier
`muse-spark-1.3-contributor`, for image-cache recovery implementation and tests,
read-only numerical-code review, and manuscript revision. Local job summaries
record the calls and executed checks; raw sessions are not part of the release.
The coordinator reviewed changes and ran integration checks. This identifier
describes the configured interface, not an independently audited backend.

## Claude editorial pass, 2026-09-21

At the author's request, the Claude CLI completed manuscript review and prose
revision using `claude-fable-5-1` (Claude Fable 5.1). The successful response
metadata records that model identifier and the `firstParty` provider. The run
received the manuscript and public methodological/citation notes, with tools
disabled; it performed no experiments and no independent literature search.
A later scoped file-editing pass used Read/Write/Edit/Bash tools to revise the
prose and verify word counts and textual invariants. The coordinator checked
and integrated the proposed language. This contribution
is disclosed separately in the manuscript acknowledgments. Raw sessions and
private reasoning are not included in the release.
