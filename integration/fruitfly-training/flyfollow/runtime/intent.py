"""Voice or typed intent -> mode_cmd (plan 2.2 row 2: offline keyword rules, no LLM).

    parse("find my water bottle") -> {"topic": "mode_cmd", "mode": "FIND", "source": "voice", "intent": "find",
                                      "target": {"cls": "bottle", "prompt": "water bottle", "height_m": 0.22, ...}}

Intents (field "intent" on every result, so the mission can tell "stop" from "stay"):
  follow -> FOLLOW    find -> FIND with target    again -> FIND with again=True and no target (mission reuses
  the last target)    stop / stay -> HOLD (in GUIDE, the mission treats "stop" as "I am at the object" -> FOLLOW)
  return -> RETURN    land -> LAND    guide -> GUIDE
Returns None when nothing matches. Target cls is the COCO name when one exists (water bottle -> bottle, phone ->
cell phone, mug -> cup), else the spoken noun as an open-vocabulary prompt (keys -> "keys", open_vocab True).
"""

from __future__ import annotations

import re

from flyfollow.runtime.geometry import CLASS_ALIASES, canonical_class, class_size_m
from flyfollow.runtime.messages import msg

COCO = frozenset((
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
))
DEFAULT_HEIGHT_M = 0.15  # unknown object: a hand-sized thing (the controller's size prior, flagged size_known False)

_FILLER = r"(?:please|can you|could you|would you|will you|hey drone|drone|ok|okay|now|then|and|help me|for me)"
_DET = r"(?:my|the|a|an|our|your|his|her|some|me)"


def _norm(text: str) -> str:
    s = text.lower().replace("’", "'")
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return " ".join(s.split())


def _strip(s: str) -> str:
    """Drop leading/trailing filler words."""
    prev = None
    while prev != s:
        prev = s
        s = re.sub(rf"^(?:{_FILLER})\s+", "", s)
        s = re.sub(rf"\s+(?:{_FILLER})$", "", s)
        s = s.strip()
    return s


# Ordered: earlier rules win. Each is (intent, mode, regex on the normalized, filler-stripped text).
_AGAIN = re.compile(r"^(?:look|search|try|check|find it)\s+again\b|^keep\s+(?:looking|searching)\b|^(?:look|search)\s+more\b"
                    r"|^try\s+(?:once\s+)?more\b")
_LAND = re.compile(r"^land\b|^(?:go\s+)?land\s+(?:now|the drone)\b|^touch\s*down\b")
_RETURN = re.compile(r"^(?:come|go|fly)\s+back\b|^return\b|^come\s+(?:here|to me)\b|^back\s+to\s+me\b")
_GUIDE = re.compile(r"^guide\b|^(?:take|lead|bring|walk)\s+me\s+(?:there|to (?:it|them))\b|^lead\s+(?:me|the way)\b"
                    r"|^show\s+me\s+(?:the\s+way|where)\b")
_FOLLOW = re.compile(r"^follow\b|^come\s+with\s+me\b|^(?:resume|keep following|continue)\b|^let'?s\s+go\b")
_STOP = re.compile(r"^(?:stop|halt|freeze|pause|stop following|don't follow)\b")
_STAY = re.compile(r"^(?:stay|wait|hold|hover|hang on|hold on|hold position|stay (?:there|here|put))\b")
_FIND = re.compile(
    rf"^(?:(?:find|locate|search for|look for|look around for|get|fetch|seek)\s+(?:{_DET}\s+)?"
    rf"|where(?:'s|'re| is| are| did i (?:put|leave))\s+(?:{_DET}\s+)?"
    rf"|i (?:lost|can't find|cannot find|dropped|need|want)\s+(?:{_DET}\s+)?"
    rf"|have you seen\s+(?:{_DET}\s+)?)(.+)$"
)
_TAIL = re.compile(r"\s+(?:please|for me|now|again|at|go|gone|is|are|i put|i left)$")


def resolve_object(phrase: str) -> dict | None:
    """Spoken object phrase -> mode_cmd target {cls, prompt, height_m, open_vocab, size_known}."""
    p = re.sub(rf"^(?:{_DET})\s+", "", phrase.strip())
    prev = None
    while prev != p:
        prev = p
        p = _TAIL.sub("", p).strip()
    if not p or p in ("it", "that", "this", "them", "something", "stuff", "things"):
        return None
    words = p.split()
    cls = None
    for n in range(len(words), 0, -1):  # longest known suffix: "blue water bottle" -> "water bottle" -> bottle
        for i in range(0, len(words) - n + 1):
            cand = " ".join(words[i:i + n])
            c = canonical_class(cand)
            if class_size_m(c) is not None or c in COCO or cand in CLASS_ALIASES:
                cls = c
                break
            if cand.endswith("s") and class_size_m(canonical_class(cand[:-1])) is not None and cand != "keys":
                cls = canonical_class(cand[:-1])  # plural: "my glasses" stays glasses (known), "cups" -> cup
                break
        if cls:
            break
    if cls is None:
        cls = p
    h = class_size_m(cls)
    return {"cls": cls, "prompt": p, "height_m": h if h is not None else DEFAULT_HEIGHT_M,
            "open_vocab": cls not in COCO, "size_known": h is not None}


def parse(text: str, source: str = "voice") -> dict | None:
    """Text -> mode_cmd message dict (topic, t, mode, source, intent, [target], [again]) or None."""
    s = _strip(_norm(text or ""))
    if not s:
        return None
    if _AGAIN.search(s):
        return msg("mode_cmd", mode="FIND", source=source, intent="again", again=True)
    if _LAND.search(s):
        return msg("mode_cmd", mode="LAND", source=source, intent="land")
    if _RETURN.search(s):
        return msg("mode_cmd", mode="RETURN", source=source, intent="return")
    if _GUIDE.search(s):
        return msg("mode_cmd", mode="GUIDE", source=source, intent="guide")
    m = _FIND.match(s)
    if m:
        tgt = resolve_object(m.group(1))
        if tgt is None:
            return msg("mode_cmd", mode="FIND", source=source, intent="again", again=True)
        return msg("mode_cmd", mode="FIND", source=source, intent="find", target=tgt)
    if _FOLLOW.search(s):
        return msg("mode_cmd", mode="FOLLOW", source=source, intent="follow")
    if _STOP.search(s):
        return msg("mode_cmd", mode="HOLD", source=source, intent="stop")
    if _STAY.search(s):
        return msg("mode_cmd", mode="HOLD", source=source, intent="stay")
    if s.upper() in ("FOLLOW", "FIND", "APPROACH", "FACE_PERSON", "OVERWATCH", "GUIDE", "RETURN", "HOLD", "LAND"):
        return msg("mode_cmd", mode=s.upper(), source=source, intent=s)
    return None


__all__ = ["COCO", "parse", "resolve_object"]
