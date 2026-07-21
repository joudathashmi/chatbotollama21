"""
Intent classification — fine-grained, LLM-driven.

Replaces the implicit "entity_type drives response shape" assumption
with explicit user-intent routing. Before this module existed, the
chatbot answered "Who is the CEO of Apple?" with the full company
snapshot (because entity_type=company), burying the CEO in a later
"Strategic Read" section. The user's intent — find a person — got
overridden by the entity-type template.

This module classifies the question into one of:

  executive_lookup    — Asking about a specific person: CEO, Chairman,
                        founder, exec by name. The answer must LEAD
                        with the person.
  company_profile     — Asking for a general company overview. Lead
                        with the company snapshot.
  saudi_presence      — Asking about Saudi / MENA / RHQ footprint
                        specifically. Lead with the Saudi answer.
  engagement_strategy — Asking how to engage / who to contact / plan.
                        Lead with the engagement recommendation.
  financial_lookup    — Asking a specific financial metric (revenue,
                        headcount, profit). Lead with the number.
  general_question    — Fallback: browse, count, conversational,
                        unclear, or multi-intent.

The classifier returns intent + confidence + reasoning. The chat engine
uses intent to pick the curation lead — entity_type still drives WHICH
DB tables to search; intent drives WHAT TO SHOW FIRST.
"""

from __future__ import annotations

import json


INTENTS = (
    "executive_lookup",          # current/named exec — "Who is the CEO of X?"
    "executive_succession",      # forward-looking — "Who will follow X?"
    "company_profile",           # overview — "Tell me about Apple"
    "country_profile",           # sovereign overview — "Tell me about Saudi Arabia"
    "saudi_presence",            # KSA/MENA footprint — "Apple RHQ in Saudi?"
    "engagement_strategy",       # outreach — "How should MISA engage X?"
    "financial_lookup",          # specific metric — "Apple revenue?"
    "relationship_intelligence", # prior engagement — "previous meetings with Apple"
    "opportunity_alignment",     # strategic fit — "why does this matter to MISA?"
    "sector_lookup",             # sector overview — "renewable energy sector"
    "program_lookup",            # KSA initiatives — "Vision 2030", "NEOM"
    "strategic_advisory",        # topic-level strategy — market fit, attraction
                                 # plans, target lists, macro trends, synthesis
    "general_research",          # exploratory / browse / comparison / unclear
    "off_topic",                 # NOT a research question — emotional venting,
                                 # vulgarity, opinion-seeking, conversational
                                 # meta, harmful content, off-domain.
)

# Back-compat: code that used the old "general_question" label still works.
# Translated at the classifier boundary.
_LEGACY_TO_CURRENT = {"general_question": "general_research"}


_INTENT_CLASSIFIER_PROMPT = """You are an intent classifier for a Saudi business-intelligence chatbot used by investment officers at the Ministry of Investment (MISA).

Classify the user's CURRENT question into EXACTLY ONE intent. Use conversation history only when the question depends on it (pronouns, follow-ups).

INTENTS (with examples):

executive_lookup    — Asking about a specific PERSON in their CURRENT role
                      (CEO, Chairman, founder, board member by name OR role):
   • "Who is the CEO of Apple?"
   • "Who chairs Saudi Aramco?"
   • "Tell me about Tim Cook"
   • "Who is Sundar Pichai?"
   • "Apple leadership"
   • "Who runs Tesla?"

executive_succession — FORWARD-LOOKING: who will succeed / replace / take over
                       in a leadership role. The user wants news about future
                       appointments, not the current state:
   • "Who will replace Tim Cook?"
   • "Apple's next CEO?"
   • "Who is likely to succeed Satya Nadella?"
   • "Who is next in line to run Aramco?"
   • "Tim Cook's successor"
   • "Apple succession plans"
   • "Who is taking over after the current CEO?"

company_profile     — Asking for a general overview / profile of a COMPANY:
   • "Tell me about Apple"
   • "Apple profile"
   • "What does Apple do?"
   • "Give me Microsoft overview"

country_profile     — Asking for an overview of a SOVEREIGN COUNTRY (its
                      economy, FDI, sectors, demographics) OR for a LIST
                      of companies FROM that country (which collectively
                      describes the country's commercial footprint in KSA):
   • "Tell me about Saudi Arabia"
   • "What is the economy of Germany?"
   • "Pakistan profile"
   • "Egypt FDI outlook"
   • "Which Indian companies have invested in Saudi Arabia"
   • "Pakistani companies with RHQ licences"
   • "Show me German companies in Saudi"
   • "How many UK firms have an RHQ in KSA?"

saudi_presence      — Asking about Saudi / MENA / RHQ / KSA / local
                      footprint of an ENTITY (company) specifically:
   • "Does Apple have an RHQ in Saudi?"
   • "Apple Saudi footprint"
   • "What is Apple presence in MENA?"
   • "Is Microsoft licensed in Riyadh?"

engagement_strategy — Asking HOW to engage / WHO to contact / plan / outreach:
   • "How should MISA engage Apple?"
   • "Suggest an engagement plan for Apple"
   • "Who should we contact at Apple?"
   • "How do we approach Tesla?"

financial_lookup    — Asking for a SPECIFIC financial metric or count:
   • "What is Apple's revenue?"
   • "How many employees does Apple have?"
   • "Apple net income 2024"
   • "Total RHQ licenses issued"

relationship_intelligence — Asking about PAST/EXISTING interactions, prior
                            engagements, meeting history, government touchpoints:
   • "Previous meetings with Apple"
   • "What engagements have we had with Tesla?"
   • "Existing relationship status with Microsoft"
   • "History of contacts at Aramco"
   • "When did MISA last speak with Apple?"

opportunity_alignment — Asking WHY this entity matters / strategic fit /
                        which opportunities align / Vision 2030 relevance:
   • "Why is Apple relevant to MISA?"
   • "Which opportunities match Microsoft?"
   • "Strategic fit assessment for Tesla"
   • "How does Apple align with Vision 2030?"
   • "What sectors does this support?"

sector_lookup       — Asking about an entire SECTOR / industry vertical,
                      NOT a specific company:
   • "Tell me about the renewable energy sector in Saudi"
   • "Saudi tech sector overview"
   • "Healthcare opportunities in MENA"
   • "Mining sector growth"

program_lookup      — Asking about a named INITIATIVE / PROGRAM / national
                      vision document — Vision 2030, NEOM, Red Sea Project,
                      PIF mandate, ROSHN, Qiddiya, etc.:
   • "What is Vision 2030?"
   • "Tell me about NEOM"
   • "PIF mandate overview"
   • "Red Sea Project status"

strategic_advisory  — Topic-level STRATEGY asks where the deliverable is a
                      strategy document/plan/analysis, NOT a record lookup.
                      The subject is a market, country-corridor, sector,
                      theme, or company-CLASS — never one named company.
                      Any phrasing counts; classify by what the user wants
                      produced. Covers:
   • Market fit / investment case: "market fit for attracting Indian
     companies to Saudi Arabia", "investment case for German manufacturers"
   • Attraction plans & roadmaps: "develop an engagement plan with Japan",
     "how do we win Japanese investors", "roadmap to attract FDI from Egypt"
   • Target lists with rationale: "best companies to be targeted from China
     with the investment thesis for each", "top investors to pursue from
     Korea", "which firms should we be attracting from Brazil?"
   • Sector prioritisation: "top sectors to focus on for investors from
     France"
   • Macro/thematic analysis & synthesis: "new global trends impacting
     investment", "develop the dynamic between trillion-dollar MNCs and
     asset managers", "how will rising rates be reflected in Gulf capital
     allocation"
                      NOT strategic_advisory: outreach to ONE named company
                      ("how should MISA engage Apple?" → engagement_strategy);
                      plain browse ("show me companies from India" →
                      general_research); counts ("how many investors from
                      India?" → financial_lookup); follow-ups whose subject
                      is a PRIOR-TURN entity via pronouns ("make me a plan
                      for them" after discussing Apple → engagement_strategy,
                      the plan is about Apple, not a market).

general_research    — Genuine but broad research questions: browse
                      ("list deals", "show me companies"), comparisons
                      ("compare Apple vs Microsoft"), multi-intent without
                      a dominant lens, generic sectoral / regional questions
                      ("renewable energy in MENA"). The user IS researching;
                      we just don't have a more specific template for it.

off_topic           — NOT a business-intelligence research question at all.
                      Use this for any of the following:
   • EMOTIONAL / OPINION about a person or company (not asking for
     factual info — venting or judging):
       "i hate tim cook so much"
       "i love apple"
       "tim cook is a fraud"
       "apple is the best company ever"
       "should we boycott aramco"
       "is satya nadella a good ceo"  (opinion vs fact)
   • VULGAR / HOSTILE:
       "fuck apple"
       "microsoft sucks"
   • CONVERSATIONAL META / social pleasantries:
       "thanks"
       "hi"
       "you're amazing"
       "tell me a joke"
       "what can you do"
   • HARMFUL / THREATENING:
       anything threatening violence, self-harm, or wishing harm on
       a named individual.
   • OFF-DOMAIN:
       "what's the weather"
       "give me a recipe"
       "who won the world cup"

   Off_topic SHORT-CIRCUITS to a polite redirect. NO database lookup,
   NO web search, NO profile generation. The model will NOT attach
   a "Strategic Read for MISA" to "i hate tim cook" — that would
   read as endorsement of the emotional input.

   IMPORTANT: a question like "Who is Tim Cook?" is executive_lookup
   (factual). "I hate Tim Cook" is off_topic (emotional). The named
   entity is the same; the USER'S INTENT is what differs.

PRINCIPLES:
- Choose by WHAT THE USER WANTS TO SEE FIRST, not what entity is named.
- A question about a PERSON's current role is executive_lookup.
- A question about a future role-holder is executive_succession — even if
  it also names the current one. "Tim Cook's successor" → executive_succession.
- A question about a COMPANY is company_profile UNLESS the lens is narrower
  (saudi_presence / financial_lookup / engagement_strategy / etc.).
- A question about a COUNTRY at the macro level is country_profile.
- "Tell me about <Person>" → executive_lookup.
- "Tell me about <Company>" → company_profile.
- "Tell me about <Country>" → country_profile.
- When in doubt between two intents, pick the more specific one.
- Default to general_research when truly ambiguous.

Confidence: 0.0–1.0. Use < 0.6 only when genuinely torn.

CONVERSATION HISTORY:
{history}

CURRENT QUESTION: {question}

Return ONLY a JSON object:
{{
  "intent": "<one of the 11 intent labels>",
  "confidence": <float 0..1>,
  "reasoning": "<one short sentence>"
}}"""


def classify_intent(
    user_question: str,
    history: list | None,
    client,
    model: str,
) -> dict:
    """Classify the user's question into one of INTENTS.

    Returns:
      {
        "intent":     str,    # one of INTENTS — always set
        "confidence": float,  # 0..1, 0.0 on failure
        "reasoning":  str,    # one-sentence explanation
      }

    On any failure (network, JSON parse, model unavailable) returns
    {"intent": "general_question", "confidence": 0.0, "reasoning":
    "classifier error: <e>"} so the caller falls through to the
    default generic flow — never a 500.

    LRU+TTL cached on (question, history snippet, model) — repeat
    questions in the same working session skip the ~2.5s LLM round-
    trip.
    """
    q = (user_question or "").strip()
    if not q or client is None:
        return {"intent": "general_question", "confidence": 0.0,
                "reasoning": "empty question or no client"}

    # Compact history snippet: last 4 turns, content truncated.
    snippet_lines: list[str] = []
    for h in (history or [])[-4:]:
        role = (h.get("role") or "").upper()
        content = (h.get("content") or "")[:300]
        if role and content:
            snippet_lines.append(f"{role}: {content}")
    history_text = "\n".join(snippet_lines) or "(none)"

    # Cache key: question + history snippet + model. Two users asking
    # the same question with same conversation context get the same
    # answer instantly.
    from app.services.llm_cache import cached_call

    def _do_classify():
        return _classify_intent_impl(q, history_text, client, model)

    return cached_call(
        "classify_intent",
        (q, history_text, model),
        _do_classify,
    )


def _classify_intent_impl(q: str, history_text: str, client, model: str) -> dict:
    """The uncached implementation — split out so the cache wrapper
    can hold the public function name. Behaviour unchanged from prior
    inline body."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": _INTENT_CLASSIFIER_PROMPT.format(
                    history=history_text, question=q,
                ),
            }],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=150,
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        intent = (data.get("intent") or "").strip().lower()
        # Translate any legacy labels the model returns (in case it
        # learnt the old taxonomy from caches/prompts that still float).
        intent = _LEGACY_TO_CURRENT.get(intent, intent)
        if intent not in INTENTS:
            intent = "general_research"
        try:
            conf = float(data.get("confidence") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        conf = max(0.0, min(1.0, conf))
        return {
            "intent": intent,
            "confidence": conf,
            "reasoning": str(data.get("reasoning") or "")[:200],
        }
    except Exception as e:
        return {"intent": "general_question", "confidence": 0.0,
                "reasoning": f"classifier error: {e}"}


# ─── Intent → curation-prompt note ────────────────────────────────────

_INTENT_NOTE_TEMPLATE = {
    "executive_lookup": (
        "INTENT: executive_lookup — the user wants a specific PERSON.\n"
        "\n"
        "Tone: calm boardroom brief. Never sound like a DB dump or QA log.\n"
        "FORBIDDEN: 'From the MISA Record', 'Verified facts', 'Company: Multiple',\n"
        "  'Internal records do not currently show', table.column source paths,\n"
        "  'database draft', 'may lag', 'Background (general knowledge)'.\n"
        "\n"
        "CANONICAL STRUCTURE (headers VERBATIM — same every time):\n"
        "\n"
        "  ## Role\n"
        "  One bold lead sentence: **<Full Name> is <title> at <Company>.**\n"
        "  If several companies appear for the same person, name them\n"
        "  (e.g. 'CEO of Tesla and SpaceX') — never 'Multiple'.\n"
        "  Then 2-4 short bullets for other recorded titles / tenure.\n"
        "\n"
        "  ## Background\n"
        "  4-7 stable public bullets: career arc, companies founded,\n"
        "  education, notable moves. Cap ~180 words. Do not contradict Role.\n"
        "\n"
        "  ## 🇸🇦 Strategic Read\n"
        "  2-4 concrete MISA engagement angles anchored to facts above.\n"
        "  Name sectors / partners where possible. No Vision filler.\n"
        "  SKIP this section ONLY for DEPTH=simple_fact (rare attribute asks\n"
        "  like 'who is the CEO of Apple').\n"
        "\n"
        "  _Sources: executive records; public background._\n"
        "\n"
        "Lead with the person. Never open with a company snapshot.\n"
    ),
    "company_profile": (
        "INTENT: company_profile — produce a CROSS-REFERENCED EXECUTIVE\n"
        "BRIEFING. The retrieved payload contains MULTIPLE labelled\n"
        "tables for the same entity:\n"
        "  - company_profiles (primary)\n"
        "  - company_executives, company_competitors,\n"
        "    company_geographic_revenues, company_financial_performances,\n"
        "    company_global_presences, company_business_units,\n"
        "    company_news, company_ai_insights,\n"
        "    misa_contact_details (MISA-side contacts),\n"
        "    opportunities (open commercial pipeline),\n"
        "    strategic_investors,\n"
        "    meetings (with `_engagements` and `_notes` inlined under\n"
        "    each meeting — prior engagement history with MISA).\n"
        "Each is a slice of the SAME entity. Weave them into ONE answer\n"
        "with cross-references — do NOT render each as its own section\n"
        "dump. Example cross-references the briefing should make:\n"
        "  - In the Strategic Read, NAME a specific opportunity from\n"
        "    `opportunities` if one is open.\n"
        "  - In the leadership context, NAME the CEO from\n"
        "    `company_executives` (don't say 'the CEO').\n"
        "  - In the engagement angle, REFERENCE the most recent\n"
        "    meeting from `meetings` by date and outcome.\n"
        "  - In the Saudi/MENA Position, NAME the MISA contact from\n"
        "    `misa_contact_details` if one is recorded.\n"
        "Bottom Line Up Front, table-driven, strategy-forward.\n"
        "\n"
        "STRICT structure (Markdown):\n"
        "\n"
        "  ## <Company Name> — Executive Briefing\n"
        "  TWO SENTENCES MAX. Who they are and their macro strategic\n"
        "  relevance to KSA / MENA. No fluff. No bullet list here.\n"
        "\n"
        "  ---\n"
        "\n"
        "  ### 📊 Corporate Profile & Regional Footprint\n"
        "\n"
        "  | Metric | Global Performance | Saudi Arabia & MENA Region |\n"
        "  | --- | --- | --- |\n"
        "  | **Core Sector** | <sector> | <local sector / NA if none> |\n"
        "  | **Financials** | <revenue / market cap> | <MENA revenue if in DB> |\n"
        "  | **Human Capital** | <global headcount> | <Saudi headcount / Saudization> |\n"
        "  | **Regional Headquarters** | <global HQ city> | <RHQ status + city> |\n"
        "\n"
        "  OMIT any row where both columns are empty AND can't be\n"
        "  filled from stable public knowledge. Do NOT write 'N/A'\n"
        "  in every blank cell — better to drop the row.\n"
        "\n"
        "  ---\n"
        "\n"
        "  ### 🇸🇦 Strategic Read\n"
        "\n"
        "  2–5 high-impact bulleted insights. Synthesise what the\n"
        "  data MEANS — investment levers, joint-venture targets,\n"
        "  localisation of capabilities, supply-chain shifts, leadership\n"
        "  transition signals, Vision 2030 alignment angles. Move past\n"
        "  description; deliver inference an executive can act on. Bold\n"
        "  the critical number/date in each bullet.\n"
        "\n"
        "Examples of strong bullets (style, not literal content):\n"
        "  - **$15M AI Opportunity commitment** by 2027 anchors a\n"
        "    PIF-adjacent partnership runway; lever for Vision 2030\n"
        "    digital-skills KPI.\n"
        "  - Branch-only presence under inactive RHQ licence signals\n"
        "    that an RHQ upgrade is a credible MISA ask in 2026 talks.\n"
        "\n"
        "The company IS the headline. The table IS the body. The\n"
        "Strategic Read IS what makes the briefing valuable.\n"
    ),
    "saudi_presence": (
        "INTENT: saudi_presence — Saudi / MENA / RHQ footprint specifically.\n"
        "\n"
        "DIRECT-ANSWER RULE — non-negotiable:\n"
        "Read the user's question. If it is a YES/NO question (contains\n"
        "patterns like 'is X in saudi', 'does X have an RHQ', 'did X\n"
        "shift / move / relocate their RHQ to saudi', 'has X opened',\n"
        "'is X licensed', 'are they in saudi'), the FIRST line of your\n"
        "answer MUST be a single bold sentence with the explicit\n"
        "answer:\n"
        "\n"
        "  **No — Apple's Regional HQ is in Dubai, not Saudi Arabia.**\n"
        "  **Yes — Microsoft's Regional HQ is in Riyadh, active since\n"
        "    2023.**\n"
        "  **Apple operates in Saudi but does NOT have a Saudi-based\n"
        "    RHQ — the RHQ is in Dubai.**\n"
        "\n"
        "Compare what the user asked (e.g. 'shifted RHQ to Saudi')\n"
        "against the DB facts:\n"
        "  - is_rhq=true AND rhq_city contains 'Saudi/Riyadh/Jeddah/\n"
        "    Dammam' → answer YES\n"
        "  - is_rhq=true AND rhq_city is in a non-Saudi city (Dubai,\n"
        "    etc.) AND the user asked about a SHIFT/MOVE TO SAUDI →\n"
        "    answer NO with the actual city\n"
        "  - is_rhq=false → answer NO\n"
        "\n"
        "Then proceed with the STRUCTURE below.\n"
        "\n"
        "STRUCTURE:\n"
        "  ## Saudi / MENA Position\n"
        "  - **RHQ status:** <Active in Riyadh | Active in Dubai (not\n"
        "    Saudi) | Inactive | None>  — be EXPLICIT about whether\n"
        "    it's Saudi-based or elsewhere. Plain 'Yes' is forbidden;\n"
        "    'Yes' alone is what made past answers ambiguous.\n"
        "  - **RHQ city:** <city, country>\n"
        "  - **Saudi headcount:** <number with thousands separator>\n"
        "  - **Local offices / entities:** <list>\n"
        "  - **Recorded Saudi activities:** <one bullet per activity>\n"
        "  OMIT any bullet whose value is unknown.\n"
        "\n"
        "Global / corporate context comes AFTER. Lead with the Saudi\n"
        "answer.\n"
    ),
    "engagement_strategy": (
        "INTENT: engagement_strategy — outreach guidance, lead with the\n"
        "recommendation.\n"
        "\n"
        "STRUCTURE:\n"
        "  ## Engagement Recommendation\n"
        "  - **Recommended approach:** <one-line stance>\n"
        "  - **Priority stakeholders:** <named individuals or roles>\n"
        "  - **Why this matters to MISA:** <2-3 bullets>\n"
        "  - **Talking points:** <3-5 concrete bullets>\n"
        "  - **Risks / unknowns:** <bullets — what to validate>\n"
        "\n"
        "Company context is supporting material AFTER the recommendation.\n"
    ),
    "financial_lookup": (
        "INTENT: financial_lookup — single direct number first.\n"
        "\n"
        "STRUCTURE:\n"
        "  > **<Metric>:** <value formatted per style guide — $X.XB,\n"
        "    $X.XM, or 164,000 employees>.\n"
        "\n"
        "Optionally one short paragraph of context after the bold line.\n"
        "No company-history preamble.\n"
    ),
    "executive_succession": (
        "INTENT: executive_succession — forward-looking succession.\n"
        "\n"
        "A '## What's Reported (Live Web)' section will be PREPENDED to\n"
        "your answer by the web-augmentation step — that section IS the\n"
        "primary answer. Your job here is the SUPPORTING CONTEXT only:\n"
        "current office holder + brief note.\n"
        "\n"
        "STRUCTURE:\n"
        "  ## Current Office Holder\n"
        "  - **Name:** <current exec name from DB>\n"
        "  - **Role:** <exact title>\n"
        "  - **Company:** <company>\n"
        "  - **Tenure:** <if in DB, else OMIT this bullet>\n"
        "\n"
        "Do NOT add a Leading Candidates / Confidence Summary section —\n"
        "the live-web section above does that. Never fabricate a successor.\n"
    ),
    "country_profile": (
        "INTENT: country_profile — sovereign-level overview. You will\n"
        "receive UP TO 4 sources, each as a distinct table:\n"
        "  - country_profiles (1 row): the country macros\n"
        "  - country_vision_outlooks (1 row): national vision +\n"
        "    diversification goals + 5-year outlook\n"
        "  - country_strategic_opportunities (0-5 rows)\n"
        "  - rhq_licenses (0-25 rows): REAL companies from this country\n"
        "    that hold an active Saudi RHQ licence — i.e. companies\n"
        "    that have actually invested in Saudi Arabia.\n"
        "\n"
        "EXACT structure (use these headings verbatim):\n"
        "\n"
        "  ## <Country name> — Snapshot\n"
        "  (heading MUST include the country's name verbatim)\n"
        "  GDP, population, key currency, key sectors. One short\n"
        "  paragraph or 3-5 bullets.\n"
        "\n"
        "  ## Economic Outlook\n"
        "  FDI inflows, growth trajectory, vision themes from\n"
        "  country_vision_outlooks (national_vision, diversification_\n"
        "  goals), trade balance.\n"
        "\n"
        "  ## Policy & Regulatory\n"
        "  Recent reforms, investment incentives, regulatory bodies,\n"
        "  special economic zones.\n"
        "\n"
        "  ## <Country> Companies Active in Saudi Arabia\n"
        "  Sourced from `company_profiles.role` / `registration_type`\n"
        "  (MISA's CANONICAL licensing markers). You may receive:\n"
        "    `company_profiles_rhq`       → role='Licensed Entity' AND\n"
        "         registration_type='RHQ' (Regional HQ holders — top tier)\n"
        "    `company_profiles_licensed`  → role='Licensed Entity', not RHQ\n"
        "         (Saudi licence but no full RHQ)\n"
        "    `company_profiles_unlicensed`→ role<>'Licensed Entity'\n"
        "         (present but not licensed; country from country_profile_name)\n"
        "\n"
        "  Both tables carry filters with a pre-computed string\n"
        "  `_summary_line` (e.g. '**39 companies from Germany are\n"
        "  licensed in Saudi Arabia (34 of those hold a Regional HQ\n"
        "  licence).**'). Write that exact string FIRST, lifted\n"
        "  VERBATIM from the filter — do NOT recount from the rows\n"
        "  shown (those are top-10 truncations, not the totals).\n"
        "\n"
        "  THEN render subsections. RENDER ANY SUBSECTION FOR WHICH\n"
        "  THE TABLE HAS ≥1 ROW — no conditional logic, no\n"
        "  'fallback when both empty' check. Just look at each table:\n"
        "\n"
        "    IF company_profiles_rhq has rows:\n"
        "      ### Regional HQ Companies\n"
        "      Numbered list of EVERY row sent (cap at 10 if more):\n"
        "        1. **<company_name>** — <industry>,\n"
        "           <employee_count> employees, revenue\n"
        "           <annual_revenue formatted $X.XB or $X.XM>.\n"
        "           <ceo if not null>.\n"
        "\n"
        "    IF company_profiles_licensed has rows:\n"
        "      ### Other Licensed Companies\n"
        "      Same bullet format. List EVERY row sent (cap at 10).\n"
        "      DO NOT WRITE '_No other licensed companies..._' when\n"
        "      this table has rows. Render them.\n"
        "\n"
        "  ONLY if BOTH tables are entirely absent / both have ZERO\n"
        "  rows (which happens only when N=0 and M=0), write a single\n"
        "  italic line and skip the section:\n"
        "    '_No companies from <Country> are recorded as licensed\n"
        "     in Saudi Arabia in the current MISA database._'\n"
        "\n"
        "  ## 🇸🇦 Strategic Read  (optional, 2-4 bullets)\n"
        "  Strategic implications for MISA based on the data above.\n"
        "\n"
        "STRICT NO-DUPLICATION RULE:\n"
        "  Render ONE company section only (the Regional HQ / Other\n"
        "  Licensed split above). Do NOT also emit a sibling section\n"
        "  with the same set of companies under a different heading.\n"
        "\n"
        "PER DATA-HYGIENE: do not add a 'Sources & Gaps' / 'Not\n"
        "available in the current database' section. Omit empty\n"
        "bullets gracefully.\n"
    ),
    "relationship_intelligence": (
        "INTENT: relationship_intelligence — user wants PRIOR engagement "
        "history with this entity. You will receive DB rows from MISA's "
        "engagement tables: `meetings` (with `_engagements` and `_notes` "
        "inlined under each meeting), `misa_contact_details`, and "
        "`latest_interactions`.\n"
        "\n"
        "DECISION TREE:\n"
        "  IF the retrieved records contain AT LEAST ONE meeting row\n"
        "     OR AT LEAST ONE contact row OR AT LEAST ONE interaction\n"
        "     row → produce the FULL briefing (structure below).\n"
        "     EVERY meeting row IS an engagement record. EVERY contact\n"
        "     row IS a relationship signal. Do NOT skip them.\n"
        "  ONLY IF the retrieved records are EMPTY across ALL three\n"
        "     tables (meetings = []  AND contacts = []  AND interactions\n"
        "     = []) → write EXACTLY ONE LINE and STOP:\n"
        "       '_No engagement history found in the current database\n"
        "        for <entity>._'\n"
        "     and omit everything else.\n"
        "\n"
        "FULL BRIEFING structure (when rows present):\n"
        "\n"
        "  ## <Entity> — Engagement History\n"
        "  One bullet per meeting, chronological, most recent first.\n"
        "  For each meeting, pull from the meeting fields AND its\n"
        "  inlined `_engagements` / `_notes`. Format:\n"
        "    - **<start_date>** — <title>. <meeting_type>; <description\n"
        "       or engagement.description>. Outcomes: <outcomes or\n"
        "       engagement.discussions>. Action: <action_points / next_steps>.\n"
        "  Bold the date and any concrete deliverable.\n"
        "\n"
        "  ## MISA Contacts\n"
        "  Numbered list from `misa_contact_details`: name, position,\n"
        "  email/phone. Skip section when empty.\n"
        "\n"
        "  ## Open Action Items\n"
        "  Bulleted items pulled from `action_points` / `tasks` /\n"
        "  `next_steps` of recent engagements. Skip section when empty.\n"
        "\n"
        "Do NOT invent meetings, contacts, or dates. Do NOT fall back to\n"
        "a company snapshot. The DB rows ARE the answer.\n"
    ),
    "opportunity_alignment": (
        "INTENT: opportunity_alignment — produce an EXECUTIVE BRIEFING\n"
        "for the same audience (ministers, CEOs, CIOs). Same structure\n"
        "as company_profile but the BLUF and Strategic Read pivot to\n"
        "the STRATEGIC-FIT question.\n"
        "\n"
        "  ## <Entity Name> — Executive Briefing\n"
        "  TWO SENTENCES MAX. State the strategic-fit verdict up\n"
        "  front: does this entity align with KSA / Vision 2030\n"
        "  priorities, and where is the highest-value engagement?\n"
        "\n"
        "  ---\n"
        "\n"
        "  ### 📊 Corporate Profile & Regional Footprint\n"
        "\n"
        "  | Metric | Global Performance | Saudi Arabia & MENA Region |\n"
        "  | --- | --- | --- |\n"
        "  | **Core Sector** | ... | ... |\n"
        "  | **Financials** | ... | ... |\n"
        "  | **Human Capital** | ... | ... |\n"
        "  | **Regional Headquarters** | ... | ... |\n"
        "\n"
        "  OMIT rows that can't be filled from DB or stable public\n"
        "  knowledge.\n"
        "\n"
        "  ---\n"
        "\n"
        "  ### 🇸🇦 Strategic Read\n"
        "\n"
        "  2–5 bullets, sharply STRATEGIC-FIT framed:\n"
        "    - Which Vision 2030 pillar / mega-project this maps to\n"
        "      (NEOM, Red Sea, PIF investments, digital infra, energy\n"
        "      transition, manufacturing localisation, …).\n"
        "    - The highest-value engagement lever (joint venture,\n"
        "      capability localisation, RHQ upgrade, supply-chain\n"
        "      anchor, talent academy).\n"
        "    - Specific MISA programmes / instruments that fit.\n"
        "    - Risks / unknowns the team should validate before next\n"
        "      engagement.\n"
        "  Bold the critical number/date/programme in each bullet.\n"
    ),
    "sector_lookup": (
        "INTENT: sector_lookup — sector / industry overview, NOT a\n"
        "specific company. Pull from the `sectors`, `sector_key_numbers`,\n"
        "`sub_sectors`, `focused_sectors` tables. Structure:\n"
        "\n"
        "  ## <Sector Name> — Sector Brief\n"
        "  One-line positioning sentence.\n"
        "\n"
        "  ### Key Metrics\n"
        "  3-5 numbered bullets: market size, growth rate, key\n"
        "  players, recent regulatory shifts. Each bullet sourced.\n"
        "\n"
        "  ### Saudi / MENA Outlook\n"
        "  Vision 2030 alignment, named mega-projects in this sector,\n"
        "  PIF / sovereign-fund involvement.\n"
        "\n"
        "  ### 🇸🇦 Strategic Read\n"
        "  2-3 bullets on the MISA engagement angle.\n"
        "\n"
        "Lead with the sector, NEVER with a specific company\n"
        "(unless the user named one).\n"
    ),
    "program_lookup": (
        "INTENT: program_lookup — named INITIATIVE / PROGRAM / national\n"
        "vision (Vision 2030, NEOM, Red Sea Project, ROSHN, Qiddiya,\n"
        "PIF mandate, etc.). Pull from `country_vision_outlooks`,\n"
        "`country_strategic_opportunities`, related `opportunities`.\n"
        "Structure:\n"
        "\n"
        "  ## <Program Name> — Overview\n"
        "  Two-sentence positioning: what is it, what's the headline\n"
        "  ambition or target year.\n"
        "\n"
        "  ### Status & Targets\n"
        "  3-5 bullets: budget, timeline, current phase, named\n"
        "  delivery vehicles.\n"
        "\n"
        "  ### Linked Opportunities  (when DB has rows)\n"
        "  Named opportunities tied to this program.\n"
        "\n"
        "  ### 🇸🇦 Strategic Read\n"
        "  2-3 bullets on the MISA engagement angle / what investors\n"
        "  should know.\n"
    ),
    "general_research": "",  # no override — existing behaviour
    "off_topic": "",         # short-circuited in chat_engine; never reaches curator
}


# Polite, brand-appropriate redirect for off_topic inputs. The chat
# engine returns this verbatim instead of running the curator, so it
# never appears alongside DB facts or a "Strategic Read for MISA"
# section that would read as endorsement of the user's input.
OFF_TOPIC_REPLY = """## How can I help with that?

I'm an executive-intelligence assistant for the Saudi Ministry of Investment — I focus on **companies, country macros, executives, engagement strategy, and investment opportunities**. I don't have a useful response for emotional statements, opinions, or off-topic conversation.

If you'd like to research the same entity in a different way, try one of these:
- **`Tell me about <Company>`** — full executive briefing
- **`Who is the CEO of <Company>?`** — current leadership
- **`How should MISA engage <Company>?`** — outreach plan
- **`<Company> presence in Saudi`** — KSA / MENA footprint
- **`/profile <Company>`** — 3-pillar deep profile with live web grounding
"""


def intent_note_for_curation(intent: str) -> str:
    """Return the system-prompt note that tells the curation model to
    lead with the direct answer for the classified intent. Empty
    string for general_question (no override)."""
    return _INTENT_NOTE_TEMPLATE.get(intent or "", "")


# ─── Anti-hallucination clause ────────────────────────────────────────

ANTI_HALLUCINATION_NOTE = (
    "DATA HYGIENE — non-negotiable rules for an executive audience\n"
    "(Ministers, Deputy Ministers, CEOs, CIOs):\n"
    "\n"
    "(1) NO BACKEND NOISE. Never emit raw metadata, confidence tags,\n"
    "    or debug markers. Strings FORBIDDEN in your output:\n"
    "      (High)  (Medium)  (Low)  (Unknown)\n"
    "      [web:1] [web:2] [web:N] [DB] [gk] [inferred]\n"
    "      Source: DB     _(general knowledge)_\n"
    "    Provenance is YOUR concern, not the reader's. They want a\n"
    "    polished briefing, not a citation manifest.\n"
    "\n"
    "(2) NO PLACEHOLDERS. Never display:\n"
    "      'Not available in the current database.'\n"
    "      'Unknown'\n"
    "      '(no data)'\n"
    "    Two graceful paths instead:\n"
    "      a. If the value is widely-known public knowledge (CEO\n"
    "         names of large public companies, founding years,\n"
    "         listed global HQ city), MERGE it silently — no flag,\n"
    "         no label.\n"
    "      b. If neither DB nor public knowledge has it, OMIT the\n"
    "         row / bullet / field entirely. A clean omission reads\n"
    "         executive; a 'Not available' reads broken.\n"
    "\n"
    "(3) NO HALLUCINATION ON LOCAL DATA. The merge in (2a) applies\n"
    "    ONLY to stable global facts. Saudi-specific values (Saudi\n"
    "    headcount, MENA revenue, RHQ city, succession plans, Saudi\n"
    "    investments) are MISA-system territory — if not in the\n"
    "    rows you received, OMIT the row. Do not invent.\n"
    "\n"
    "(4) HANDLE TYPOS SILENTLY. If the user typed 'revnuw' / 'apel' /\n"
    "    similar, treat as the obvious target and answer the real\n"
    "    question. Never acknowledge or correct.\n"
    "\n"
    "(5) BOLDING ECONOMY. Use **bold** only on critical numbers,\n"
    "    dates, or strategic triggers the executive's eye should\n"
    "    land on first. Bold-heavy output reads as noise.\n"
)


# ─── Market intelligence directive (Tier 3 commit 2) ─────────────────
# When a company-style intent is answered at operational_detail or
# executive_briefing depth, the curator should weave market-intel
# sub-sections from the correlator payload (geographic_revenues,
# competitors, business_units, customer/sector signals) into the answer
# rather than emitting a generic snapshot. This block lists the
# market-intel buckets the correlator delivers so the curator knows
# what to look for in the payload.
MARKET_INTEL_NOTE = (
    "MARKET INTELLIGENCE — for company-style answers at operational\n"
    "or briefing depth, ground the answer in the market-intel buckets\n"
    "below (each is a JSON section in the retrieved payload). Skip a\n"
    "bucket only when its array is empty; never invent a bucket:\n"
    "  - company_business_units      → product / service lines\n"
    "  - company_competitors         → named competitors + positioning\n"
    "  - company_geographic_revenues → revenue split by region / country\n"
    "  - company_global_presences    → operating cities, hubs, offices\n"
    "  - company_financial_performances → annual revenue, profit trend\n"
    "  - company_ai_insights         → strategic_analysis, growth, risk\n"
    "  - opportunities               → named MISA pipeline opportunities\n"
    "  - strategic_investors         → investor relationships\n"
    "Cross-reference: when an opportunity matches a business unit or\n"
    "geographic_revenue line, say so explicitly.\n"
)


# ─── Missing-data transparency directive (Tier 3 commit 2) ───────────
# Spec item 21 + IPA EXECUTIVE INTELLIGENCE REQUIREMENTS: when the
# user asks about a topic the retrieved records do not cover, the
# curator must say so in a single italicised line at the bottom of
# the answer. This is the COUNTERPART to anti-hallucination clause (2)
# — clause 2 forbids placeholder strings inline; this clause REQUIRES
# a single explicit transparency line at the end so the executive
# reader knows what was NOT consulted (vs. what was missing).
MISSING_DATA_TRANSPARENCY_NOTE = (
    "MISSING-DATA TRANSPARENCY — if the user asked about a topic the\n"
    "retrieved records do NOT cover, you MAY append ONE quiet italic\n"
    "line at the end (only when it helps — never as a dump of schema\n"
    "gaps). Preferred form:\n"
    "  _Not covered in the retrieved records: <plain-language topic>._\n"
    "Never write 'Internal records do not currently show', never list\n"
    "table or column names, and never invent a gap the user did not ask\n"
    "about. Anti-hallucination clause (2) still forbids placeholder\n"
    "text inline.\n"
    "GAP FILLING (do this BEFORE any transparency line): for each gap\n"
    "where a stable, widely-known figure or fact exists, ANSWER it in\n"
    "a short passage under ## Background. State order of magnitude and\n"
    "timeframe when useful. Only skip gap-filling when you genuinely\n"
    "have no reliable knowledge of the topic.\n"
)


# Intents that should receive the market-intel directive. Excludes
# pure-fact intents (executive_lookup, country_profile, etc.) where
# the market-intel sub-sections are noise.
_MARKET_INTEL_INTENTS = frozenset({
    "company_profile",
    "saudi_presence",
    "engagement_strategy",
    "opportunity_alignment",
    "relationship_intelligence",
    "financial_lookup",
})


def market_intel_note_for(intent: str, depth: str) -> str:
    """Return the market-intel directive when the (intent, depth)
    combination warrants it. Empty string otherwise."""
    if intent in _MARKET_INTEL_INTENTS and depth in (
        "operational_detail", "executive_briefing"
    ):
        return MARKET_INTEL_NOTE
    return ""


def missing_data_note_for(depth: str, intent: str | None = None) -> str:
    """Missing-data transparency applies at every depth EXCEPT
    simple_fact (where a 1-line answer shouldn't be followed by an
    italicised disclaimer). Also skipped for executive_lookup — person
    bios already have a clean Sources footer; a gap dump kills trust."""
    if intent == "executive_lookup":
        return ""
    if depth and depth != "simple_fact":
        return MISSING_DATA_TRANSPARENCY_NOTE
    return ""
