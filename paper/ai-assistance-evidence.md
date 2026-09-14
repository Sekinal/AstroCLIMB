# AI-assistance evidence for author disclosure

AI assistants supported programming, debugging, experiment coordination, documentation and manuscript preparation. The authors remain responsible for the scientific claims, implementation, citations and submitted text. This note distinguishes recorded execution from model names mentioned in requests; it is an evidence inventory, not a claim that every generated contribution was correct or retained.

| Model name | Evidence status | Scope and limitation |
|---|---|---|
| GLM-5.3-Flash | Recorded executions | The project-specific ZCode model-usage metadata records5,830 completed requests,36 cancelled and19 errors under this exact model identifier. These are model requests, not independent experiments or accepted code changes. |
| GLM-5.3 | Recorded executions | The same metadata records3 completed requests under this exact identifier. Its use is therefore supported independently of user requests. |
| DeepSeek Flash | Recorded launcher configuration | Later implementation-run receipts identify `deepseek-flash` and, in some cases, `deepseek-official/deepseek-flash`. These establish configured backend identifiers. This audit did not verify a provider response identifying an exact “V4.1” revision, so that version suffix remains unconfirmed. |
| GPT-6 Astra | Reported assistance; exact runtime attribution incomplete in this audit | Historical operational messages attribute reviews and a separate stricter campaign to an assistant named Astra. That name alone does not prove an exact model revision. The ZCode usage table contains only the GLM identifiers above; separate orchestration evidence is needed to bind the reported GPT-6 Astra model name to individual contributions. |

Stored backend identifiers do not independently attest a provider's internal weights or version mapping. Requesting a model, naming a model in a handoff, and recording a completed provider request are different kinds of evidence. The manuscript should use the level of specificity supported by the records and describe substantive assistance, rather than presenting a model roster as verification of the research.

No private reasoning, credentials, or raw conversation excerpts are included here. Detailed accounting remains in the private provenance archive. The recorded request counts cover the historical ZCode project sessions inspected, not the entire multi-tool campaign.
