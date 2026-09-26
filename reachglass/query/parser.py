"""Text query -> constrained intent + target class.

    parser = KeywordQueryParser()
    q = parser.parse("Can you find my water bottle?", vocabulary=["bottle", "chair", "person"])
    q.intent == "find", q.target == "bottle"

The target is ALWAYS one of `vocabulary` (the classes our detectors can actually find): the parser
never invents an object. If the user asks for something we cannot detect, intent is "find" with
target None and `unknown_target` says what was asked, so the mission can answer "I can't look for keys yet".

Intents: find | cancel | land | follow | describe | unknown
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

INTENTS = ("find", "cancel", "land", "follow", "describe", "unknown")

# canonical class -> phrases that name it (longest phrases are matched first)
SYNONYMS: dict[str, list[str]] = {
    "bottle": ["water bottle", "bottle of water", "bottle", "flask", "thermos", "hydro flask", "canteen",
               "something to drink", "my drink", "a drink", "water"],
    "cup": ["coffee cup", "cup", "mug", "glass of water", "tumbler"],
    "chair": ["somewhere to sit", "a seat", "chair", "seat", "stool"],
    "couch": ["couch", "sofa"],
    "bed": ["bed"],
    "dining table": ["dining table", "kitchen table", "table", "desk"],
    "backpack": ["backpack", "back pack", "rucksack", "school bag", "bag"],
    "handbag": ["handbag", "purse"],
    "laptop": ["laptop", "computer", "macbook"],
    "cell phone": ["cell phone", "cellphone", "mobile phone", "smartphone", "phone"],
    "remote": ["remote control", "tv remote", "remote"],
    "book": ["book", "notebook"],
    "keyboard": ["keyboard"],
    "tv": ["television", "tv", "monitor", "screen"],
    "person": ["a person", "someone", "somebody", "person"],
    "refrigerator": ["refrigerator", "fridge"],
    "sink": ["sink"],
    "toilet": ["toilet", "bathroom", "restroom"],
    "door": ["door", "exit", "way out"],
    "suitcase": ["suitcase", "luggage"],
    "umbrella": ["umbrella"],
    "scissors": ["scissors"],
    "potted plant": ["plant"],
}

# needs expressed as purposes -> candidate classes, in preference order
PURPOSES: list[tuple[str, list[str]]] = [
    (r"\bthirsty\b|\bdrink\b|\bhydrat", ["bottle", "cup"]),
    (r"\bsit\b|\bsit down\b|\btired\b|\brest\b", ["chair", "couch", "bench", "bed"]),
    (r"\bget out\b|\bleave\b|\bexit\b|\bway out\b", ["door"]),
    (r"\bcall\b|\btext\b", ["cell phone"]),
    (r"\bsleep\b|\blie down\b", ["bed", "couch"]),
    (r"\bwork\b|\bemail\b", ["laptop"]),
]

INTENT_PATTERNS: list[tuple[str, str]] = [
    ("land", r"\bland\b|\bcome down\b|\btouch down\b|\bshut down\b"),
    ("cancel", r"\bcancel\b|\bstop\b|\babort\b|\bnever ?mind\b|\bforget (it|about)\b"),
    ("describe", r"what'?s around|what is around|what do you see|what can you see|describe|look around"),
    ("follow", r"\bfollow me\b|\bcome back\b|\bcome here\b|\breturn\b|\bfollow\b"),
    ("find", r"\bfind\b|\bwhere\b|\blook(ing)? for\b|\bsearch\b|\blocate\b|\bget me\b|\btake me\b|\bguide me\b|"
             r"\bbring me\b|\bneed\b|\bwant\b|\bshow me\b|\blost\b|\bcan'?t see\b"),
]


@dataclass
class ParsedQuery:
    text: str
    intent: str
    target: str | None = None
    confidence: float = 0.0
    unknown_target: str | None = None  # something asked for that is not in the vocabulary
    reason: str = ""


def normalize(text: str) -> str:
    t = text.lower().replace("’", "'")
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


class QueryParser(ABC):
    @abstractmethod
    def parse(self, text: str, vocabulary: list[str]) -> ParsedQuery: ...


class KeywordQueryParser(QueryParser):
    """Deterministic, offline, instant. Handles direct requests and common purposes."""

    def __init__(self, synonyms: dict[str, list[str]] | None = None):
        self.synonyms = synonyms or SYNONYMS
        phrases = [(p, cls) for cls, ps in self.synonyms.items() for p in ps]
        self._phrases = sorted(phrases, key=lambda x: -len(x[0]))

    def _mentioned(self, t: str) -> list[tuple[int, str, str]]:
        """(position, class, phrase) for every class phrase in the text, longest phrases win overlaps."""
        taken = [False] * len(t)
        found = []
        for phrase, cls in self._phrases:
            for m in re.finditer(r"\b" + re.escape(phrase) + r"\b", t):
                if not any(taken[m.start():m.end()]):
                    for i in range(m.start(), m.end()):
                        taken[i] = True
                    found.append((m.start(), cls, phrase))
        return sorted(found)

    def parse(self, text: str, vocabulary: list[str]) -> ParsedQuery:
        t = normalize(text)
        vocab = set(vocabulary)
        if not t:
            return ParsedQuery(text, "unknown", reason="empty")
        hits = {name for name, pat in INTENT_PATTERNS if re.search(pat, t)}
        mentioned = self._mentioned(t)
        detectable = [m for m in mentioned if m[1] in vocab]
        # priority: land > find (a detectable object named with a find verb) > cancel > describe > follow > find
        if "land" in hits:
            return ParsedQuery(text, "land", confidence=0.9, reason="keyword 'land'")
        if not ("find" in hits and detectable):
            for name in ("cancel", "describe", "follow"):
                # "look around for my bottle" / "follow me to the chair": an object makes it a find
                if name in hits and not (name in ("follow", "describe") and detectable):
                    return ParsedQuery(text, name, confidence=0.9, reason=f"keyword '{name}'")
        intent = "find" if ("find" in hits or "describe" in hits) else None
        # prefer the object named right after the verb ("forget the cup, find my bottle" -> bottle)
        verb = None
        for m in re.finditer(INTENT_PATTERNS[-1][1], t):
            verb = m.end()
        if verb is not None:
            after = [m for m in mentioned if m[0] >= verb]
            mentioned = after + [m for m in mentioned if m[0] < verb]
        # a named object: first one we can detect; otherwise remember what was asked
        for _, cls, phrase in mentioned:
            if cls in vocab:
                return ParsedQuery(text, "find", cls, 0.9 if intent == "find" else 0.75, reason=f"'{phrase}' -> {cls}")
        if mentioned:
            _, cls, phrase = mentioned[0]
            return ParsedQuery(text, "find", None, 0.8, unknown_target=cls, reason=f"'{phrase}' -> {cls} (not detectable)")
        # a purpose ("I'm thirsty")
        for pat, cands in PURPOSES:
            if re.search(pat, t):
                for cls in cands:
                    if cls in vocab:
                        return ParsedQuery(text, "find", cls, 0.6, reason=f"purpose -> {cls}")
                return ParsedQuery(text, "find", None, 0.5, unknown_target=cands[0], reason="purpose, nothing detectable")
        if intent == "find":
            m = re.search(r"\b(?:find|where(?: is| are|'s)?|look(?:ing)? for|search for|locate|get me)\s+(?:my |the |a |an |some )?(.+)$", t)
            what = m.group(1) if m else None
            return ParsedQuery(text, "find", None, 0.5, unknown_target=what, reason="object not recognised")
        return ParsedQuery(text, "unknown", reason="no intent or object recognised")


class LLMQueryParser(QueryParser):
    """Asks a language model, but only accepts answers inside the vocabulary; falls back to keywords.

    `complete(prompt) -> str` is any function that returns the model's text (Grok, Claude, local...),
    so this class has no network code of its own and is testable offline.
    """

    PROMPT = (
        "You map a blind user's spoken request to a JSON command for an assistive drone.\n"
        "Allowed intents: find, cancel, land, follow, describe, unknown.\n"
        "For find, target MUST be exactly one of: {vocab} (choose the object that best serves the request, "
        "e.g. 'I'm thirsty' -> bottle), or null if none fits.\n"
        'Reply with only JSON: {{"intent": ..., "target": ..., "confidence": 0..1}}\n'
        "Request: {text}"
    )

    def __init__(self, complete: Callable[[str], str], fallback: QueryParser | None = None):
        self.complete = complete
        self.fallback = fallback or KeywordQueryParser()

    def parse(self, text: str, vocabulary: list[str]) -> ParsedQuery:
        fb = self.fallback.parse(text, vocabulary)
        try:
            raw = self.complete(self.PROMPT.format(vocab=", ".join(sorted(vocabulary)), text=text))
            m = re.search(r"\{.*\}", raw, re.S)
            data = json.loads(m.group(0)) if m else {}
            intent = str(data.get("intent", "")).lower()
            target = data.get("target")
            conf = float(data.get("confidence", 0.5))
        except Exception as e:  # network/parse problems must never block the mission
            fb.reason = f"llm failed ({e.__class__.__name__}); {fb.reason}"
            return fb
        if intent not in INTENTS:
            return fb
        if intent == "find":
            if target not in vocabulary:  # hallucinated or undetectable object: do not trust it
                return fb if fb.target else ParsedQuery(text, "find", None, conf, unknown_target=target, reason="llm target not detectable")
            return ParsedQuery(text, "find", target, conf, reason="llm")
        return ParsedQuery(text, intent, None, conf, reason="llm")
