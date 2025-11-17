"""
Text processing utilities for conversation management.

Ported from backend-orchestration text processing modules.
"""

import logging
from typing import Optional, Set, Tuple

logger = logging.getLogger(__name__)


class TextContext:
    """
    Extracts meaningful text segments (contexts) from a given string.

    Identifies substrings that end with predefined split tokens and adhere
    to specified length constraints.
    """

    def __init__(self, split_tokens: Optional[Set[str]] = None):
        """
        Initialize text context processor.

        Args:
            split_tokens: Set of characters that mark context boundaries
        """
        if split_tokens is None:
            self.split_tokens: Set[str] = {".", "!", "?", ",", ";", ":", "\n", "-", "。", "、"}
        else:
            self.split_tokens: Set[str] = set(split_tokens)

    def get_context(
        self, txt: str, min_len: int = 6, max_len: int = 120, min_alnum_count: int = 10
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Find the shortest valid context at the beginning of the input text.

        Args:
            txt: Input string
            min_len: Minimum context length
            max_len: Maximum context length
            min_alnum_count: Minimum alphanumeric character count

        Returns:
            Tuple of (context string, remaining string) or (None, None)
        """
        alnum_count = 0

        for i in range(1, min(len(txt), max_len) + 1):
            char = txt[i - 1]
            if char.isalnum():
                alnum_count += 1

            # Check if current character is a context end
            if char in self.split_tokens:
                # Check if criteria are met
                if i >= min_len and alnum_count >= min_alnum_count:
                    context_str = txt[:i]
                    remaining_str = txt[i:]
                    logger.debug(f"Context found at position {i}: {context_str}")
                    return context_str, remaining_str

        return None, None


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate simple text similarity based on word overlap.

    Args:
        text1: First text string
        text2: Second text string

    Returns:
        Similarity score between 0.0 and 1.0
    """
    # Simple word-based similarity
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 or not words2:
        return 0.0

    intersection = words1.intersection(words2)
    union = words1.union(words2)

    return len(intersection) / len(union) if union else 0.0
