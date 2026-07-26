# MISA Intelligence API — Release Notes

**Build period:** 17 to 26 July 2026
**Repository:** github.com/joudathashmi/chatbotollama21
**Canonical folder:** NIMs_Chatbot_ollama21

This build brings the local model, the data residency controls, the security
hardening, the document safeguards, and every response quality fix into a single
deployable codebase. Grounded data now stays on the machine by default via a local
Llama 3.1 model, uploads are screened for classification and consent before anything
is stored, and each answer type has been graded against the reference briefings.

**At a glance:** 1,119 automated tests passing, ruff lint clean, four classification
levels gated, one consolidated code line.

Legend: **New** capability, **Fixed** defect, **Improved** existing behaviour.

## 1. Local AI and data residency

Grounded intelligence is composed on a local model so database and document content
never leaves the host.

| Feature | What it does | Status |
| --- | --- | --- |
| Local model (Ollama) | Data grounded prompts are composed by a local `llama3.1` model over an OpenAI compatible endpoint. No database or document text is sent to a cloud provider for these calls. | New |
| Strict residency mode | A residency policy layer classifies every model call as question only or data grounded, and routes grounded calls to the local model. A guard blocks any call that would send grounded data to a cloud model. | New |
| Narrative quality path | Optional mode that sends only privacy filtered fact cards (never raw rows) to Azure for full briefing quality. Turn it off for hard local only operation. | New |
| Determinism controls | Temperature zero and a fixed seed so the same question reproduces the same answer. Response cache is now opt in and off by default. | Improved |

## 2. Security and data protection

Internal data is stripped before any model sees it, and the service refuses to boot
in production with an unsafe configuration.

| Feature | What it does | Status |
| --- | --- | --- |
| Privacy filter | Internal comment fields (reviewer notes, MISA and team comments, review status, creator and audit metadata) are stripped from every row before it reaches any model. Only public fields survive. | New |
| PII masking | Emails and phone numbers are masked before prompting. | New |
| Prompt injection guard | Attempts to override instructions or extract the system prompt or schema are refused before any model or database work runs. | New |
| Authentication and roles | JWT login and refresh, named accounts with bcrypt hashes, role based access, and refresh token revocation on every protected route. | New |
| Production boot gate | Refuses to start in production when auth is disabled, CORS is open, the malware scanner is off, docs are exposed, or the JWT secret is weak. | New |
| Malware scanner accuracy | The PDF JavaScript heuristic now parses the document structure instead of matching raw bytes, ending false positives on large graphic heavy reports while still rejecting real threats. | Fixed |

## 3. Document handling

Only Public documents can enter the library, consent is required, and retrieval is
robust to how a question is phrased.

| Feature | What it does | Status |
| --- | --- | --- |
| Classification gate | Detects the classification itself. Screens text, DOCX headers, footers and text boxes, the filename, and hidden labels written by classifier tools (Boldon James, Microsoft Purview) for Restricted, Secret, Top Secret and Confidential markings in English and Arabic. Marked files are rejected before storage. | New |
| Consent declaration | Every upload must accept a consent declaration (Public classification, no political or offensive content, PDPL personal data limits, upload rights, processing consent). Served by a versioned policy endpoint and logged per acceptance. | New |
| Retroactive sweep | A maintenance script re-screens the whole library with the current detector and removes any marked document carried over from before the gate existed. | New |
| Upload size limits | Document and request body caps are now configurable (raised to forty megabytes) so large graphic heavy reports upload and index. | Improved |
| Retrieval routing | Document search now matches on any relevant term rather than requiring every word, so a question finds the right passage regardless of phrasing. Relevance gates keep an unrelated document from pre empting a database or advisory answer. | Fixed |

## 4. Response quality

Every answer type was regenerated live and graded against the reference briefings for
specificity, not generic prose.

| Feature | What it does | Status |
| --- | --- | --- |
| Intelligence Briefing format | The established briefing layout is restored across company profiles, engagement recommendations, and company targeting: footprint counts, named RHQ and licensed tables, the investment thesis matrix, and counterpart anchored recommendations. | Fixed |
| Adaptive strategy analysis | Abstract analytical questions now take a thematic shape instead of being forced into the country corridor template with irrelevant sector tables and roadmaps. | Fixed |
| RHQ license holder lists | Questions like which companies hold an RHQ license return a real, revenue ranked holder table instead of a general knowledge fallback. | New |
| Strategic Read discipline | Each bullet must be a defensible record fact, a named MISA action, or an attraction implication. For a fund or state vehicle it names the companies it capitalises as the real targets. | Improved |
| Revenue ranking | Company lists are ranked by true numeric revenue. Previously text sorting placed a small restaurant above a multi billion bank. | Fixed |
| Filler removal | Hollow adjectives (world class, cutting edge) are removed by a grammar safe scrub, leaving tighter, more specific prose. | Improved |

## 5. Platform and consolidation

Parallel work streams are merged into one version controlled, deployable codebase.

| Feature | What it does | Status |
| --- | --- | --- |
| Single consolidated repo | The local model line and the security and prompt line, previously in separate folders, are merged into one git repository with both feature sets and no lost work. | New |
| Verified test suite | The full suite passes at 1,119 tests with ruff lint clean, the CI hard gate. Auth tests are hermetic and no longer depend on local settings. | Improved |
| Deployment change log | A shareable change log document ships in the repository with the full environment variable checklist for the deployment team. | New |

---

_Prepared 26 July 2026. Backends: local Llama 3.1 and Azure OpenAI (privacy filtered)._
