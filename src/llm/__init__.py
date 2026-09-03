"""
LLM inference and candidate generation package.
"""

from .candidate_generator import ActionCandidate, generate_legal_candidates
from .ollama_client import OllamaClient
from .response_parser import ParsedPlan, parse_model_response
from .next_action_contract import (
    NEXT_ACTION_SYSTEM_PROMPT,
    NextActionParse,
    build_next_action_prompt,
    format_next_action_completion,
    parse_next_action_response,
)

__all__ = [
    "ActionCandidate",
    "generate_legal_candidates",
    "OllamaClient",
    "ParsedPlan",
    "parse_model_response",
    "NEXT_ACTION_SYSTEM_PROMPT",
    "NextActionParse",
    "build_next_action_prompt",
    "format_next_action_completion",
    "parse_next_action_response",
]
