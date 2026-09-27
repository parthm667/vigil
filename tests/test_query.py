"""Step 6: query parsing."""

import json

import pytest

from reachglass.query import QUERY_PARSERS, KeywordQueryParser, LLMQueryParser

VOCAB = ["bottle", "person", "chair", "couch", "dining table", "cup", "backpack", "laptop", "cell phone"]
P = KeywordQueryParser()


@pytest.mark.parametrize("text,target", [
    ("Find my water bottle", "bottle"),
    ("Can you find my water bottle please?", "bottle"),
    ("Where's my bottle?", "bottle"),
    ("where is the water", "bottle"),
    ("I'm thirsty", "bottle"),
    ("I need something to drink", "bottle"),
    ("Hey, I lost my Hydro Flask somewhere around here, can you look for it?", "bottle"),
    ("Where did I leave my phone", "cell phone"),
    ("take me to my backpack", "backpack"),
    ("I need somewhere to sit", "chair"),
    ("I'm tired, I want to sit down", "chair"),
    ("Find the sofa", "couch"),
    ("Find my laptop", "laptop"),
    ("stop following me and find my bottle", "bottle"),
    ("FIND MY MUG!!", "cup"),
])
def test_find_requests(text, target):
    q = P.parse(text, VOCAB)
    assert (q.intent, q.target) == ("find", target), q.reason
    assert q.confidence > 0.4


def test_target_must_be_in_vocabulary():
    q = P.parse("find my water bottle", ["person", "chair"])  # bottle model not loaded
    assert q.intent == "find" and q.target is None and q.unknown_target == "bottle"
    q = P.parse("I'm thirsty", ["person", "cup"])
    assert q.target == "cup"  # falls to the next candidate that IS detectable
    q = P.parse("take me to the door", VOCAB)
    assert q.target is None and q.unknown_target == "door"


def test_unknown_object_is_reported_not_invented():
    q = P.parse("Can you find my keys?", VOCAB)
    assert q.intent == "find" and q.target is None and q.unknown_target == "keys"


@pytest.mark.parametrize("text,intent", [
    ("stop", "cancel"), ("cancel that", "cancel"), ("never mind", "cancel"), ("forget it", "cancel"),
    ("land", "land"), ("please land the drone now", "land"), ("stop and land", "land"),
    ("follow me", "follow"), ("come back", "follow"),
    ("what's around me?", "describe"), ("What do you see", "describe"),
    ("hello there", "unknown"), ("", "unknown"), ("   ", "unknown"),
])
def test_other_intents(text, intent):
    assert P.parse(text, VOCAB).intent == intent


def test_longest_phrase_wins():
    q = P.parse("find the water bottle near the table", VOCAB)
    assert q.target == "bottle"  # 'water bottle' (not 'water' + 'table'): first detectable mention


def test_llm_parser_accepts_valid_answers_only():
    def model(answer):
        return lambda prompt: answer

    ok = LLMQueryParser(model('{"intent": "find", "target": "chair", "confidence": 0.8}'))
    q = ok.parse("I'd like to rest my legs", VOCAB)
    assert (q.intent, q.target, q.reason) == ("find", "chair", "llm")
    # hallucinated target -> falls back to keywords (which found nothing detectable) and reports it
    bad = LLMQueryParser(model('Sure! {"intent": "find", "target": "golden retriever", "confidence": 0.9}'))
    q = bad.parse("find my dog", VOCAB)
    assert q.target is None and q.unknown_target == "golden retriever"
    # hallucinated target but keywords know better
    q = LLMQueryParser(model('{"intent": "find", "target": "flask"}')).parse("find my water bottle", VOCAB)
    assert q.target == "bottle"
    # garbage / exceptions -> keyword fallback
    q = LLMQueryParser(model("I cannot help")).parse("find my bottle", VOCAB)
    assert q.target == "bottle"

    def boom(prompt):
        raise TimeoutError

    q = LLMQueryParser(boom).parse("find my bottle", VOCAB)
    assert q.target == "bottle" and "llm failed" in q.reason
    q = LLMQueryParser(model(json.dumps({"intent": "land"}))).parse("bring it down", VOCAB)
    assert q.intent == "land"


def test_llm_prompt_contains_vocabulary():
    seen = {}

    def model(prompt):
        seen["p"] = prompt
        return '{"intent": "unknown"}'

    LLMQueryParser(model).parse("hi", ["bottle", "chair"])
    assert "bottle, chair" in seen["p"] and "hi" in seen["p"]


def test_registry():
    assert isinstance(QUERY_PARSERS.build("keyword"), KeywordQueryParser)


# ------------------------------------------------------------------ enrolled names ("find arthur")
def test_enrolled_name_parses_as_find_target():
    p = KeywordQueryParser()
    vocab = ["bottle", "chair", "arthur"]
    q = p.parse("find arthur", vocab)
    assert (q.intent, q.target) == ("find", "arthur")
    q = p.parse("Where's Arthur?", vocab)
    assert (q.intent, q.target) == ("find", "arthur")
    q = p.parse("help me find the person named arthur", vocab)
    assert (q.intent, q.target) == ("find", "arthur")
    # names not enrolled stay unknown
    q = p.parse("find nathan", vocab)
    assert q.intent == "find" and q.target is None
    # the thing named after the find verb wins
    q = p.parse("forget arthur, find my bottle", vocab)
    assert q.target == "bottle"
    q = p.parse("forget the bottle, find arthur", vocab)
    assert q.target == "arthur"
