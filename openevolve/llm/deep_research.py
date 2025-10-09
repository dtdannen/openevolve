"""
OpenAI Deep Research client for system prompt augmentation
"""

import logging
import os
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


class DeepResearchClient:
    """Client for OpenAI Deep Research API"""

    def __init__(
        self,
        model: str = "o3-deep-research-2025-06-26",
        timeout: int = 600,
        max_tokens: int = 16000,
        api_key: Optional[str] = None,
    ):
        """
        Initialize Deep Research client

        Args:
            model: Deep Research model to use
            timeout: Maximum time to wait for research (seconds)
            max_tokens: Maximum tokens in research output
            api_key: OpenAI API key (uses env var if None)
        """
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens

        # Initialize OpenAI client
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

        self.client = OpenAI(api_key=api_key, timeout=timeout)
        logger.info(f"Initialized Deep Research client with model: {model}")

    def research(self, query: str) -> str:
        """
        Execute a deep research query (blocking)

        Args:
            query: Research query/prompt

        Returns:
            Research results as string

        Raises:
            Exception: If research fails or times out
        """
        logger.info(f"Starting Deep Research query (timeout: {self.timeout}s)")
        logger.debug(f"Query: {query[:200]}...")

        try:
            # Call Deep Research API using the responses endpoint
            response = self.client.responses.create(
                model=self.model,
                messages=[{"role": "user", "content": query}],
                max_tokens=self.max_tokens,
            )

            # Extract the research results
            if response.choices and len(response.choices) > 0:
                result = response.choices[0].message.content
                logger.info(
                    f"Deep Research completed successfully ({len(result)} characters)"
                )
                return result
            else:
                raise ValueError("No response from Deep Research API")

        except Exception as e:
            logger.error(f"Deep Research failed: {e}")
            raise

    def research_with_fallback(self, query: str, fallback_message: str = "") -> str:
        """
        Execute research with fallback on failure

        Args:
            query: Research query
            fallback_message: Message to return if research fails

        Returns:
            Research results or fallback message
        """
        try:
            return self.research(query)
        except Exception as e:
            logger.warning(f"Deep Research failed, using fallback: {e}")
            return fallback_message or "Deep Research unavailable - continuing with current prompt."
