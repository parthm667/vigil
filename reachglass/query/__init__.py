from ..registry import Registry
from .parser import INTENTS, KeywordQueryParser, LLMQueryParser, ParsedQuery, QueryParser, normalize

QUERY_PARSERS = Registry("query parser")
QUERY_PARSERS.register("keyword")(KeywordQueryParser)

__all__ = ["QUERY_PARSERS", "QueryParser", "KeywordQueryParser", "LLMQueryParser", "ParsedQuery", "INTENTS", "normalize"]
