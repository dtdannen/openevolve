"""
System prompt augmenter using Deep Research and global learnings
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional

from openevolve.config import DeepResearchConfig
from openevolve.database import ProgramDatabase
from openevolve.llm.deep_research import DeepResearchClient
from openevolve.llm.ensemble import LLMEnsemble
from openevolve.prompt.templates import TemplateManager

logger = logging.getLogger(__name__)


class SystemPromptAugmenter:
    """Augments system prompts using Deep Research and evolution learnings"""

    def __init__(
        self,
        config: DeepResearchConfig,
        output_dir: str,
        llm_ensemble: LLMEnsemble,
        language: str = "python",
    ):
        """
        Initialize the system prompt augmenter

        Args:
            config: Deep research configuration
            output_dir: Output directory for this run
            llm_ensemble: LLM ensemble for meta-prompting
            language: Programming language being evolved
        """
        self.config = config
        self.output_dir = output_dir
        self.llm_ensemble = llm_ensemble
        self.language = language

        # Set up prompts directory (run-specific)
        self.prompts_dir = os.path.join(output_dir, "augmented_system_prompts")
        os.makedirs(self.prompts_dir, exist_ok=True)
        logger.info(f"Initialized system prompt augmenter: {self.prompts_dir}")

        # Initialize Deep Research client
        self.deep_research_client = DeepResearchClient(
            model=config.model,
            timeout=config.timeout,
            max_tokens=config.max_tokens,
        )

        # Load templates
        self.template_manager = TemplateManager()

        # Track current iteration/version
        self.current_version = 0

    def save_prompt_version(self, prompt: str, iteration: int) -> None:
        """
        Save a prompt version to disk

        Args:
            prompt: Prompt text to save
            iteration: Iteration number (used as filename)
        """
        version_path = os.path.join(self.prompts_dir, f"{iteration}.txt")
        with open(version_path, "w") as f:
            f.write(prompt)
        logger.info(f"Saved prompt version {iteration} to {version_path}")
        self.current_version = iteration

    def get_current_prompt(self) -> Optional[str]:
        """
        Get the most recent prompt version

        Returns:
            Latest prompt text, or None if no versions exist
        """
        if self.current_version == 0:
            return None

        version_path = os.path.join(self.prompts_dir, f"{self.current_version}.txt")
        if os.path.exists(version_path):
            with open(version_path, "r") as f:
                return f.read()
        return None

    def collect_global_learnings_stub(self, database: ProgramDatabase) -> Dict[str, str]:
        """
        Collect global learnings from evolution (STUB for now)

        Args:
            database: Program database

        Returns:
            Dictionary with learning summaries
        """
        # TODO: Replace with actual global learnings when available
        best_program = database.get_best_program()
        total_programs = len(database.programs)

        stub_summary = f"""
Evolution Statistics:
- Total programs evaluated: {total_programs}
- Best program score: {best_program.metrics.get('combined_score', 0):.4f} (found at iteration {best_program.iteration_found})
- Number of islands: {database.config.num_islands}
- Language: {self.language}

Note: Detailed global learnings integration pending - this is a placeholder summary.
"""
        return {
            "summary": stub_summary.strip(),
            "total_programs": str(total_programs),
            "best_score": f"{best_program.metrics.get('combined_score', 0):.4f}",
        }

    def build_research_query(
        self, learnings: Dict[str, str], base_prompt: str, iteration: int
    ) -> str:
        """
        Build Deep Research query from learnings and context

        Args:
            learnings: Global learnings dictionary
            base_prompt: Current base system prompt
            iteration: Current iteration number

        Returns:
            Formatted research query
        """
        # Load template
        try:
            query_template = self.template_manager.get_template("deep_research_query")
        except Exception as e:
            logger.warning(f"Could not load deep_research_query template: {e}, using default")
            query_template = """Research Query: How can we improve code evolution strategies for {language} programs?

Current Context:
- Iteration: {iteration} of evolution completed
- Global Learnings: {learnings_summary}

Base System Prompt Being Used:
{base_prompt}

Please research best practices, common pitfalls, and optimization strategies for evolving {language} code that align with our current findings.

Focus on actionable insights that can be added to our system prompt to improve future iterations."""

        # Format the query
        query = query_template.format(
            language=self.language,
            iteration=iteration,
            learnings_summary=learnings.get("summary", "No learnings available"),
            base_prompt=base_prompt[:1000],  # Truncate to avoid excessive length
        )

        return query

    async def augment_prompt(
        self, base_prompt: str, research_results: str, learnings: Dict[str, str], iteration: int
    ) -> str:
        """
        Generate augmentation section using meta-prompting

        Args:
            base_prompt: Original base prompt
            research_results: Results from Deep Research
            learnings: Global learnings summary
            iteration: Current iteration

        Returns:
            New augmented prompt with research insights appended
        """
        # Load meta-prompt template
        try:
            meta_template = self.template_manager.get_template("augmentation_meta_prompt")
        except Exception as e:
            logger.warning(f"Could not load augmentation_meta_prompt template: {e}, using default")
            meta_template = """You are a meta-optimizer improving system prompts for code evolution.

Original Base Prompt:
---
{base_prompt}
---

Deep Research Results:
---
{research_results}
---

Global Learnings (iteration {iteration}):
{learnings_summary}

Task: Generate a NEW SECTION to APPEND to the base prompt containing insights from deep research.

Requirements:
- Concise (max 200 words)
- Use bullet points or numbered lists
- Do NOT repeat existing content from base prompt
- Focus on actionable guidance

Output format:
## Deep Research Insights (Added at iteration {iteration}):
[your insights here]"""

        # Format meta-prompt
        meta_prompt = meta_template.format(
            base_prompt=base_prompt,
            research_results=research_results[:3000],  # Truncate if too long
            learnings_summary=learnings.get("summary", ""),
            iteration=iteration,
        )

        # Generate augmentation using LLM
        logger.info("Generating prompt augmentation using LLM meta-prompting...")
        augmentation = await self.llm_ensemble.generate(
            prompt=meta_prompt,
            temperature=0.7,
            max_tokens=500,
        )

        # Combine base prompt with augmentation
        augmented_prompt = f"{base_prompt}\n\n{augmentation}"

        return augmented_prompt

    async def run_augmentation(
        self, iteration: int, database: ProgramDatabase, current_prompt: str
    ) -> str:
        """
        Run full augmentation pipeline

        Args:
            iteration: Current iteration number
            database: Program database
            current_prompt: Current system prompt

        Returns:
            Augmented system prompt
        """
        logger.info(f"🔬 Starting Deep Research augmentation at iteration {iteration}")

        try:
            # Step 1: Collect global learnings (stub for now)
            learnings = self.collect_global_learnings_stub(database)
            logger.info("Collected global learnings (stub)")

            # Step 2: Build research query
            research_query = self.build_research_query(learnings, current_prompt, iteration)
            logger.info(f"Built research query ({len(research_query)} chars)")

            # Step 3: Execute Deep Research (blocking)
            research_results = self.deep_research_client.research_with_fallback(
                research_query,
                fallback_message="Deep Research service unavailable. Continuing without augmentation.",
            )
            logger.info(f"Received Deep Research results ({len(research_results)} chars)")

            # Check if we got a real response or fallback
            if "Deep Research service unavailable" in research_results or "Deep Research unavailable" in research_results:
                logger.warning("Deep Research failed, skipping augmentation")
                return current_prompt

            # Step 4: Generate augmentation
            augmented_prompt = await self.augment_prompt(
                current_prompt, research_results, learnings, iteration
            )
            logger.info(f"Generated augmented prompt ({len(augmented_prompt)} chars)")

            # Step 5: Save new version
            self.save_prompt_version(augmented_prompt, iteration)

            logger.info(f"✅ System prompt augmented successfully at iteration {iteration}")
            return augmented_prompt

        except Exception as e:
            logger.error(f"Augmentation failed: {e}", exc_info=True)
            logger.warning("Continuing with current prompt due to augmentation failure")
            return current_prompt
