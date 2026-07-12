"""Slot-level fact checker (§5.1) — the evaluation core of the project.

For every (generation, structured record) pair, extract factual claims from
the text with regex + lexicons and verify each against the record and
cumulative match state. Design principle: **checker precision first** — a
claim is only extracted when the phrasing is confidently about THIS ball,
because a fact-checker that hallucinates violations is worse than none.
Known recall gaps are documented inline and measured separately as omissions.

This is v2. v1 was adversarially reviewed (EXPERIMENTS.md entry
"factcheck-v1-review") and produced wrong verdicts on everyday commentary:
requirement talk ("they need two more"), team/partnership milestones,
"six wickets in hand" as a boundary, bowler figures "2 for 45" as a score
claim, absence phrasing ("ten balls without a boundary"), fielder mentions on
dot balls, prospective maidens, and idioms ("caught in two minds", "stumped
for answers", "it falls safe", "that's gone, way into the stands"). v2 adds:

- requirement-context guard: numbers inside "need/needed/required/to win"
  windows are never scoring claims;
- convention-aware score checking: "45 for 2" and "2 for 45" both accepted
  when they match (score, wickets) in either order; figures-like mismatches
  (small/large) are skipped rather than guessed;
- milestone claims must be PERSONAL ("his fifty", "Kohli's fifty", "fifty
  for Kohli") — team/partnership/required fifties are not this-ball claims;
- boundary claims exclude possessives/counts ("his four", "four wickets")
  and absence windows; plural forms ("sixes") count as mentions for omission
  purposes but never as single-ball claims;
- attribution flags an uninvolved squad surname only in ACTOR position
  (followed by an action verb, or "by X" credit on a wicket claim) — naming
  the cover fielder on a dot ball is normal commentary, not a hallucination;
- maiden claims only on completed overs (prospective talk is unchecked);
- per-slot precision is booked on the slot of the originating CLAIM
  (violation["claim_slot"]), so extras/dismissal-kind violations can no
  longer corrupt other slots' precision.

Adversarial coverage lives in tests/test_factcheck.py (60+ cases, including
every v1 review repro as a regression test).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .base import unavailable

# --------------------------------------------------------------------- types


@dataclass
class Claim:
    slot: str
    value: object
    span: str  # surface evidence, for error analysis


@dataclass
class CheckResult:
    claims: list[Claim] = field(default_factory=list)
    violations: list[dict] = field(default_factory=list)
    omissions: list[str] = field(default_factory=list)


_WORD_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}

_WORDS_RE = re.compile(r"[a-z']+")


def _words_before(text: str, start: int, k: int) -> list[str]:
    return _WORDS_RE.findall(text[:start].lower())[-k:]


def _words_after(text: str, end: int, k: int) -> list[str]:
    return _WORDS_RE.findall(text[end:].lower())[:k]


_NEED_WORDS = {"need", "needs", "needed", "needing", "require", "requires",
               "required", "want", "wants", "wanting"}
_NEED_AFTER = {"needed", "required", "shy", "win", "victory", "wanted"}


def _requirement_context(text: str, start: int, end: int) -> bool:
    """True when a number is part of runs-required talk, not runs scored."""
    return bool(
        set(_words_before(text, start, 4)) & _NEED_WORDS
        or set(_words_after(text, end, 4)) & _NEED_AFTER
    )


# ------------------------------------------------------------- run claims

_NO_RUN_RE = re.compile(r"\bno runs?\b(?!\s+out)", re.I)
_DOT_RE = re.compile(r"\bdot ball\b", re.I)
# "single" as the one-run noun; never "single-handedly", "every single ...",
# or adjective uses like "a single boundary/delivery"
_SINGLE_RE = re.compile(r"(?<!every )\bsingle\b(?!-)", re.I)
_SINGLE_NOUN_AFTER = {"boundary", "boundaries", "wicket", "wickets", "ball",
                      "balls", "delivery", "over", "overs", "mistake", "shot",
                      "fielder", "moment", "man"}
_DIGIT_RUNS_RE = re.compile(r"\b(\d+)\s+(?:runs?|more)\b", re.I)
_WORD_RUNS_RE = re.compile(
    r"\b(one|two|three|five)\s+(?:runs?|more)\b|\bthey cross for (two|three)\b", re.I
)

# ---------------------------------------------------------- boundary claims

_BOUNDARY_EXCLUDE_AFTER = (
    r"(?:wides?|byes?|leg\b|wickets?\b|overs?\b|balls?\b|slips?\b|fielders?\b|"
    r"runs?\s+(?:needed|required)|needed\b|required\b|from\b)"
)
_SIX_RE = re.compile(
    rf"(?<!his )(?<!her )\bsix\b(?!\s+{_BOUNDARY_EXCLUDE_AFTER})", re.I
)
_MAXIMUM_RE = re.compile(r"\bmaximum\b", re.I)
_FOUR_RE = re.compile(
    rf"(?<!his )(?<!her )\bfour\b(?!\s+{_BOUNDARY_EXCLUDE_AFTER})", re.I
)
_BOUNDARY_RE = re.compile(r"\bboundary\b", re.I)
_BOUNDARY_MENTION_RE = re.compile(r"\bsix(?:es)?\b|\bfours?\b|\bboundar(?:y|ies)\b|\bmaximum\b", re.I)
_ABSENCE_WORDS = {"no", "not", "without", "denied", "barely"}
_EXTRAS_VALUED_RE = re.compile(r"\b(four|six|five|\d)\s+(wides?|byes?|leg byes?)\b", re.I)

# ------------------------------------------------------------ wicket claims

_PRAISE_BEFORE_BOWLED = {"well", "beautifully", "superbly", "nicely", "tidily", "brilliantly"}
_ARTICLE_AFTER_BOWLED = {"a", "an", "the", "his", "another", "one", "two", "out"}
_NEGATORS = {"no", "not", "nearly", "almost"}
_GONE_FLIGHT_WORDS = {"stands", "rope", "crowd", "miles", "way", "distance",
                      "boundary", "rows", "orbit"}

_KIND_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("caught", re.compile(
        r"\bcaught\b(?!\s+(?:up\b|in two minds|napping|on the crease|off guard|out of position))",
        re.I)),
    ("bowled", re.compile(r"\bbowled\b|\bclean(?:s|ed) him up\b|\bcastled?\b", re.I)),
    ("lbw", re.compile(r"\blbw\b|\bleg before\b", re.I)),
    ("run out", re.compile(r"\brun[- ]out\b(?!\s+(?:chance|attempt|opportunity))", re.I)),
    ("stumped", re.compile(r"\bstumped\b(?!\s+for\s+(?:answers|words|ideas|options))", re.I)),
]
_GENERIC_WICKET_RE = re.compile(
    # "he's/that's out" but never the six idiom "out of here"
    r"\bis out\b|\bgiven out\b|\bout!|\w+'s out\b(?!\s+of\b)|\bdeparts\b"
    r"|\bdismissed\b|\bhas to go\b|\bfalls\b|\bgone\b\s*[!.,]"
    r"|\bwicket\b(?!s\b)(?!\s*(?:in hand|left|remaining|keeper))",
    re.I,
)

# ------------------------------------------------------- score & milestones

_SCORE_RE = re.compile(r"\b(\d{1,3})\s*(?:/|\s+for\s+)\s*(\d{1,3})\b")
_PERSONAL_SCORE_RE = re.compile(
    r"\b(?:out|departs|falls)\s+for\s+(\d{1,3}|a duck)\b", re.I
)
# milestones must be personal: "his fifty", "Kohli's fifty", "fifty for Kohli".
# The milestone words are case-insensitive via inline (?i:...) groups while the
# name anchor [A-Z][a-z]+'s stays case-sensitive so "That's" cannot pose as a
# possessive name.
_FIFTY_WORDS = r"(?i:fifty|half[- ]century)"
_HUNDRED_WORDS = r"(?i:century|hundred|ton)"
_FIFTY_PERSONAL_RE = re.compile(
    rf"\b(?i:his|her)\s+(?i:maiden\s+)?{_FIFTY_WORDS}\b(?!\s+(?i:partnership|stand))"
    rf"|\b[A-Z][a-z]+'s\s+(?i:maiden\s+)?{_FIFTY_WORDS}\b(?!\s+(?i:partnership|stand))"
    rf"|\b{_FIFTY_WORDS}\s+(?i:for)\s+(?=[A-Z])(?!(?:The|His|Her)\b)"
)
_HUNDRED_PERSONAL_RE = re.compile(
    rf"\b(?i:his|her)\s+(?i:maiden\s+)?{_HUNDRED_WORDS}\b(?!\s+(?i:partnership|stand))"
    rf"|\b[A-Z][a-z]+'s\s+(?i:maiden\s+)?{_HUNDRED_WORDS}\b(?!\s+(?i:partnership|stand))"
    rf"|\b(?i:magnificent century)\b"
    rf"|\b{_HUNDRED_WORDS}\s+(?i:for)\s+(?=[A-Z])(?!(?:The|His|Her)\b)"
)
# "maiden fifty/wicket/T20 ..." are career-firsts, not maiden overs
_MAIDEN_RE = re.compile(
    r"\bmaiden\b(?!\s+(?:fifty|half[- ]century|century|hundred|ton|wicket|t20|odi))", re.I
)

_CAP_TOKEN_RE = re.compile(r"\b[A-Z][a-z]+\b")

# verbs that assert someone ACTED on this ball (batting, bowling, dismissing)
_ACTOR_VERBS = {
    "drives", "drove", "hits", "smashes", "smashed", "launches", "launched",
    "pulls", "pulled", "cuts", "edges", "edged", "glances", "glanced",
    "sweeps", "swept", "flicks", "flicked", "defends", "defended", "blocks",
    "blocked", "swings", "heaves", "slogs", "scampers", "takes", "took",
    "works", "worked", "nudges", "nudged", "tucks", "tucked", "punches",
    "punched", "lofts", "lofted", "hammers", "hammered", "crunches",
    "carves", "carved", "strikes", "struck", "bowls", "traps", "trapped",
    "removes", "removed", "dismisses", "dismissed", "castles", "departs",
    "whips", "whipped", "guides", "guided", "steers", "steered", "clips",
    "clipped", "plays", "played", "charges", "misses", "missed",
}


def _surname(name: str) -> str:
    return name.split()[-1] if name.strip() else ""


def _negated(text: str, start: int, window: int = 3) -> bool:
    return bool(set(_words_before(text, start, window)) & _NEGATORS)


# ---------------------------------------------------------------- extraction


def extract_claims(text: str) -> list[Claim]:
    claims: list[Claim] = []

    # --- runs
    if _NO_RUN_RE.search(text) or _DOT_RE.search(text):
        claims.append(Claim("runs", 0, "no run/dot ball"))
    for m in _SINGLE_RE.finditer(text):
        if _negated(text, m.start(), window=2):
            continue  # "not a single ..."
        after = _words_after(text, m.end(), 1)
        if after and after[0] in _SINGLE_NOUN_AFTER:
            continue  # adjective use: "a single boundary"
        claims.append(Claim("runs", 1, "single"))
        break
    for m in _DIGIT_RUNS_RE.finditer(text):
        if not _requirement_context(text, m.start(), m.end()):
            claims.append(Claim("runs", int(m.group(1)), m.group(0)))
    for m in _WORD_RUNS_RE.finditer(text):
        if _requirement_context(text, m.start(), m.end()):
            continue
        word = (m.group(1) or m.group(2)).lower()
        claims.append(Claim("runs", _WORD_NUM[word], m.group(0)))

    # --- extras with explicit value ("four leg byes"): checked as extras,
    # and shields the number word from being read as a boundary below
    extras_spans: list[tuple[int, int]] = []
    for m in _EXTRAS_VALUED_RE.finditer(text):
        word = m.group(1).lower()
        value = int(word) if word.isdigit() else _WORD_NUM[word]
        claims.append(Claim("extras", value, m.group(0)))
        extras_spans.append(m.span())

    def _shielded(pos: int) -> bool:
        return any(a <= pos < b for a, b in extras_spans)

    def _boundary_hit(pattern: re.Pattern) -> bool:
        for m in pattern.finditer(text):
            if _shielded(m.start()):
                continue
            if _requirement_context(text, m.start(), m.end()):
                continue
            if set(_words_before(text, m.start(), 4)) & _ABSENCE_WORDS:
                continue
            return True
        return False

    # --- boundaries (singular claims only; plurals are aggregate talk and
    # count as mentions for the omission check, never as this-ball claims)
    six_hit = _boundary_hit(_SIX_RE) or bool(_MAXIMUM_RE.search(text))
    if six_hit:
        claims.append(Claim("boundary", 6, "six/maximum"))
    if _boundary_hit(_FOUR_RE):
        claims.append(Claim("boundary", 4, "four"))
    if not six_hit and _boundary_hit(_BOUNDARY_RE):
        claims.append(Claim("boundary", "any", "boundary"))

    # --- wickets: specific kinds first
    for kind, pattern in _KIND_PATTERNS:
        for m in pattern.finditer(text):
            if _negated(text, m.start()):
                continue
            if kind == "bowled" and m.group(0).lower() == "bowled":
                before = _words_before(text, m.start(), 1)
                after = _words_after(text, m.end(), 2)
                if before and before[0] in _PRAISE_BEFORE_BOWLED:
                    continue
                if after and after[0] in _ARTICLE_AFTER_BOWLED:
                    continue
                # "bowled him!" is a dismissal; "bowled him a bouncer" is not
                if after and after[0] in ("him", "her") and len(after) > 1:
                    continue
            claims.append(Claim("wicket", kind, m.group(0)))
            break  # one claim per kind is enough

    if not any(c.slot == "wicket" for c in claims):
        for m in _GENERIC_WICKET_RE.finditer(text):
            if _negated(text, m.start()):
                continue
            surface = m.group(0).lower()
            if surface.startswith("gone") and (
                set(_words_after(text, m.end(), 4)) & _GONE_FLIGHT_WORDS
            ):
                continue  # ball-flight idiom: "gone, way into the stands"
            if surface == "falls":
                after = _words_after(text, m.end(), 2)
                before = _words_before(text, m.start(), 1)
                if (after and after[0] in ("safe", "short", "harmlessly", "just")) or (
                    before and before[0] in ("it", "ball")
                ):
                    continue  # the BALL falling, not the batter
            claims.append(Claim("wicket", "unspecified", m.group(0)))
            break

    # --- team score: accept both scoreboard conventions (runs-first "45/2"
    # and wickets-first "2/45"); values that fit neither convention cleanly
    # (e.g. bowler figures) are handled at verification time
    for m in _SCORE_RE.finditer(text):
        claims.append(Claim("score", (int(m.group(1)), int(m.group(2))), m.group(0)))

    for m in _PERSONAL_SCORE_RE.finditer(text):
        raw = m.group(1).lower()
        claims.append(Claim("personal_score", 0 if raw == "a duck" else int(raw), m.group(0)))

    # --- milestones (personal only, see module docstring)
    if _FIFTY_PERSONAL_RE.search(text):
        claims.append(Claim("milestone", 50, "fifty"))
    if _HUNDRED_PERSONAL_RE.search(text):
        claims.append(Claim("milestone", 100, "century"))
    for m in _MAIDEN_RE.finditer(text):
        if not _requirement_context(text, m.start(), m.end()):
            claims.append(Claim("milestone", "maiden", "maiden"))
            break

    return claims


# --------------------------------------------------------------- verification


def _violation(claim_slot: str, display_slot: str, claimed, actual, evidence) -> dict:
    return {
        "slot": display_slot,       # what kind of error (figure 3 axis)
        "claim_slot": claim_slot,   # which claim slot it counts against
        "claimed": claimed,
        "actual": actual,
        "evidence": evidence,
    }


def check_generation(text: str, record: dict) -> CheckResult:
    """Verify one generation against its structured record (a dict in the
    row["record"] shape)."""
    result = CheckResult(claims=extract_claims(text))
    runs_ok = {record["runs_total"], record["runs_batter"]}

    for claim in result.claims:
        if claim.slot == "runs" and claim.value not in runs_ok:
            result.violations.append(
                _violation("runs", "runs", claim.value, record["runs_total"], claim.span)
            )
        elif claim.slot == "extras" and claim.value != record["extras"]:
            result.violations.append(
                _violation("extras", "extras", claim.value, record["extras"], claim.span)
            )
        elif claim.slot == "boundary":
            actual = record["runs_batter"]
            if claim.value == "any":
                # byes/overthrows legitimately reach the rope
                ok = actual in (4, 6) or record["runs_total"] >= 4
            else:
                ok = actual == claim.value
            if not ok:
                result.violations.append(
                    _violation("boundary", "boundary", claim.value, actual, claim.span)
                )
        elif claim.slot == "wicket":
            if not record["wicket"]:
                result.violations.append(
                    _violation("wicket", "wicket", claim.value, "no wicket", claim.span)
                )
            elif claim.value not in ("unspecified", record["wicket_type"]):
                result.violations.append(
                    _violation("wicket", "dismissal_type", claim.value,
                               record["wicket_type"], claim.span)
                )
        elif claim.slot == "score":
            a, b = claim.value
            actual = (record["team_score"], record["team_wickets"])
            if (a, b) == actual or (b, a) == actual:
                continue
            if a > b and b <= 10:
                # confidently runs-first phrasing that mismatches the board
                result.violations.append(
                    _violation("score", "score", f"{a}/{b}",
                               f"{actual[0]}/{actual[1]}", claim.span)
                )
            # else: figures-like/ambiguous ("2 for 45") — not a team-score claim
        elif claim.slot == "personal_score":
            if claim.value != record["batter_runs_after"]:
                result.violations.append(
                    _violation("personal_score", "personal_score", claim.value,
                               record["batter_runs_after"], claim.span)
                )
        elif claim.slot == "milestone":
            if claim.value == "maiden":
                over_runs = record.get("completed_over_runs")
                # prospective maiden talk mid-over is unchecked (over_runs None)
                if over_runs is not None and over_runs != 0:
                    result.violations.append(
                        _violation("milestone", "milestone", "maiden over",
                                   over_runs, claim.span)
                    )
            elif record["batter_runs_after"] < int(claim.value):
                result.violations.append(
                    _violation("milestone", "milestone", claim.value,
                               record["batter_runs_after"], claim.span)
                )

    # --- attribution: an uninvolved squad surname in ACTOR position
    involved = {
        _surname(n)
        for n in [record["batter"], record["bowler"], record["non_striker"],
                  record["player_out"], *record.get("fielders", [])]
        if n
    }
    squad_surnames = {_surname(n) for n in record.get("squad", [])} - involved
    wicket_claimed = any(c.slot == "wicket" for c in result.claims)
    for m in _CAP_TOKEN_RE.finditer(text):
        token = m.group(0)
        if token not in squad_surnames:
            continue
        after = _words_after(text, m.end(), 2)
        acted = bool(set(after[:2]) & _ACTOR_VERBS)
        credited = (
            wicket_claimed and _words_before(text, m.start(), 1) == ["by"]
        )
        if acted or credited:
            result.claims.append(Claim("attribution", token, token))
            result.violations.append(
                _violation(
                    "attribution", "attribution", token,
                    f"not involved (batter={_surname(record['batter'])}, "
                    f"bowler={_surname(record['bowler'])})", token,
                )
            )
    # involved names asserted as actors are correct attribution claims —
    # counted so attribution precision has a denominator
    for m in _CAP_TOKEN_RE.finditer(text):
        token = m.group(0)
        if token in involved and set(_words_after(text, m.end(), 2)[:2]) & _ACTOR_VERBS:
            result.claims.append(Claim("attribution", token, token))

    # --- omissions of salient events (§5.1); plural/aggregate boundary talk
    # counts as a mention
    if record["wicket"] and not wicket_claimed:
        result.omissions.append("wicket unmentioned")
    boundary_mentioned = bool(_BOUNDARY_MENTION_RE.search(text))
    if record["runs_batter"] == 6 and not boundary_mentioned:
        result.omissions.append("six unmentioned")
    if record["runs_batter"] == 4 and not boundary_mentioned:
        result.omissions.append("four unmentioned")
    return result


# ----------------------------------------------------------------- spaCy NER


def _load_spacy(model_name: str):
    try:
        import spacy

        return spacy.load(model_name)
    except ImportError as err:
        raise LookupError(f"spacy not installed: {err}") from err
    except OSError as err:
        raise LookupError(
            f"spacy model {model_name!r} not downloadable/installed: {err}"
        ) from err


def _ner_violations(nlp, text: str, record: dict) -> list[dict]:
    """Out-of-squad PERSON entities = invented players (the regex layer can
    only catch in-squad misattributions). Active only when the spaCy model is
    installed; precision caveats documented in the dataset card."""
    squad = {n for name in record.get("squad", []) for n in name.split()}
    out = []
    for ent in nlp(text).ents:
        if ent.label_ != "PERSON":
            continue
        if not any(part in squad for part in ent.text.split()):
            out.append(
                _violation("attribution", "attribution", ent.text,
                           "no such player in squad", ent.text)
            )
    return out


# --------------------------------------------------------------- aggregation


def factcheck_metrics(rows: list[dict], generations: list[str], cfg: dict) -> dict:
    """§5.1 aggregate metrics over aligned (row, generation) pairs.

    Per-slot precision uses the CLAIM slot of each violation, so a wrong
    dismissal kind lowers wicket precision and a wrong extras value lowers
    extras precision. ``violations_by_type`` counts the display type
    (dismissal_type separate from wicket) for the by-slot figure.
    """
    if len(rows) != len(generations):
        raise ValueError("rows and generations must be aligned")
    if not rows:
        raise ValueError("factcheck_metrics needs at least one row")

    ner_mode = cfg.get("spacy_ner", "auto")
    nlp = None
    ner_meta: dict = {"mode": "off"}
    if ner_mode in ("auto", "require"):
        try:
            nlp = _load_spacy(cfg.get("spacy_model", "en_core_web_sm"))
            ner_meta = {"mode": "spacy"}
        except LookupError as err:
            if ner_mode == "require":
                raise
            ner_meta = unavailable(str(err))

    slot_claims: dict[str, int] = {}
    slot_false: dict[str, int] = {}
    violations_by_type: dict[str, int] = {}
    n_hallucinated = 0
    salient_rows = 0
    omitted_rows = 0
    wicket_rows = wicket_claimed_rows = 0
    boundary_rows = boundary_claimed_rows = 0
    example_violations: list[dict] = []
    example_omissions: list[dict] = []

    for row, text in zip(rows, generations):
        record = row["record"]
        res = check_generation(text, record)
        if nlp is not None:
            res.violations.extend(_ner_violations(nlp, text, record))

        for claim in res.claims:
            slot_claims[claim.slot] = slot_claims.get(claim.slot, 0) + 1
        for violation in res.violations:
            claim_slot = violation.get("claim_slot", violation["slot"])
            slot_false[claim_slot] = slot_false.get(claim_slot, 0) + 1
            violations_by_type[violation["slot"]] = (
                violations_by_type.get(violation["slot"], 0) + 1
            )
        if res.violations:
            n_hallucinated += 1
            if len(example_violations) < 10:
                example_violations.append({"text": text, "violations": res.violations})

        salient = record["wicket"] or record["runs_batter"] in (4, 6)
        if salient:
            salient_rows += 1
            if res.omissions:
                omitted_rows += 1
                if len(example_omissions) < 10:
                    example_omissions.append({"text": text, "omissions": res.omissions})
        if record["wicket"]:
            wicket_rows += 1
            if any(c.slot == "wicket" for c in res.claims):
                wicket_claimed_rows += 1
        if record["runs_batter"] in (4, 6):
            boundary_rows += 1
            if any(c.slot == "boundary" for c in res.claims):
                boundary_claimed_rows += 1

    n = len(rows)
    per_slot = {
        slot: {
            "claims": slot_claims.get(slot, 0),
            "false_claims": min(slot_false.get(slot, 0), slot_claims.get(slot, 0)),
            "precision": (
                round(1 - min(slot_false.get(slot, 0), slot_claims[slot]) / slot_claims[slot], 4)
                if slot_claims.get(slot) else None
            ),
        }
        for slot in sorted(set(slot_claims) | set(slot_false))
    }
    hallucination_rate = n_hallucinated / n
    return {
        "n": n,
        "hallucination_rate": round(hallucination_rate, 4),
        "omission_rate": round(omitted_rows / salient_rows, 4) if salient_rows else None,
        "faithfulness_score": round(1 - hallucination_rate, 4),
        "per_slot": per_slot,
        "violations_by_type": violations_by_type,
        "recall": {
            "wicket_mentioned": round(wicket_claimed_rows / wicket_rows, 4) if wicket_rows else None,
            "boundary_mentioned": round(boundary_claimed_rows / boundary_rows, 4) if boundary_rows else None,
        },
        "ner": ner_meta,
        "examples": {"violations": example_violations, "omissions": example_omissions},
    }
