"""
Global learnings system for OpenEvolve

Aggregates and tracks common failures and successful patterns across all islands
and iterations to provide insights that help avoid repeated mistakes.
"""

import json
import logging
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from openevolve.config import GlobalLearningsConfig

logger = logging.getLogger(__name__)


@dataclass
class FailurePattern:
    """Represents a failure pattern observed during evolution"""

    pattern_type: str  # "syntax", "runtime", "performance_regression"
    description: str
    count: int = 1
    first_seen: int = 0  # iteration number
    last_seen: int = 0
    example_error: Optional[str] = None
    # Code context for LLM summarization
    parent_code: Optional[str] = None
    child_code: Optional[str] = None
    code_diff: Optional[str] = None
    parent_metrics: Optional[Dict[str, float]] = None
    child_metrics: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FailurePattern":
        return cls(**data)


@dataclass
class SuccessPattern:
    """Represents a successful pattern observed during evolution"""

    description: str
    count: int = 1
    avg_improvement: float = 0.0
    first_seen: int = 0
    last_seen: int = 0
    # Code context for LLM summarization
    parent_code: Optional[str] = None
    child_code: Optional[str] = None
    code_diff: Optional[str] = None
    parent_metrics: Optional[Dict[str, float]] = None
    child_metrics: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SuccessPattern":
        return cls(**data)


class GlobalLearnings:
    """
    Tracks and aggregates learnings from evolution across all islands and iterations
    """

    def __init__(self, config: GlobalLearningsConfig, llm_ensemble=None):
        self.config = config
        self.failure_patterns: Dict[str, FailurePattern] = {}
        self.success_patterns: Dict[str, SuccessPattern] = {}
        self.iteration_history: List[int] = []  # Track which iterations we've seen
        self.last_update_iteration: int = 0
        self.llm_ensemble = llm_ensemble  # For generating summaries
        self.summary_cache: Dict[str, str] = {}  # Cache LLM summaries

        logger.info(f"Initialized GlobalLearnings (enabled={config.enabled})")

    def update_from_iteration(
        self,
        iteration: int,
        result: Any,
        parent_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Update learnings from an iteration result

        Args:
            iteration: Current iteration number
            result: Iteration result containing child_program, metrics, artifacts, etc.
            parent_metrics: Parent program metrics for comparison
        """
        if not self.config.enabled:
            return

        self.iteration_history.append(iteration)

        # Trim history to window size
        if len(self.iteration_history) > self.config.window_size:
            self.iteration_history = self.iteration_history[-self.config.window_size :]

        # Track failures
        if self.config.track_failures or self.config.track_both:
            self._track_failures(iteration, result)

        # Track successes
        if self.config.track_successes or self.config.track_both:
            self._track_successes(iteration, result, parent_metrics)

        self.last_update_iteration = iteration

    def _track_failures(self, iteration: int, result: Any) -> None:
        """Track failure patterns from iteration result"""
        artifacts = getattr(result, "artifacts", None)
        if not artifacts:
            return

        # Extract code context
        child_program = getattr(result, "child_program", None)
        parent = getattr(result, "parent", None)
        child_code = getattr(child_program, "code", None) if child_program else None
        parent_code = getattr(parent, "code", None) if parent else None
        code_diff = getattr(child_program, "metadata", {}).get("changes", None) if child_program else None
        child_metrics = getattr(result, "child_metrics", None)
        parent_metrics = getattr(parent, "metrics", None) if parent else None

        # Extract syntax errors
        if self.config.include_syntax_errors:
            syntax_errors = self._extract_syntax_errors(artifacts)
            for error_desc in syntax_errors:
                self._add_failure_pattern(
                    "syntax", error_desc, iteration, error_desc,
                    parent_code=parent_code, child_code=child_code, code_diff=code_diff,
                    parent_metrics=parent_metrics, child_metrics=child_metrics
                )

        # Extract runtime errors
        if self.config.include_runtime_errors:
            runtime_errors = self._extract_runtime_errors(artifacts)
            for error_desc in runtime_errors:
                self._add_failure_pattern(
                    "runtime", error_desc, iteration, error_desc,
                    parent_code=parent_code, child_code=child_code, code_diff=code_diff,
                    parent_metrics=parent_metrics, child_metrics=child_metrics
                )

        # Track performance regressions
        if self.config.include_performance_regressions:
            if child_metrics and parent and hasattr(parent, "metrics"):
                regressions = self._detect_performance_regressions(
                    parent.metrics, child_metrics
                )
                for regression_desc in regressions:
                    self._add_failure_pattern(
                        "performance_regression", regression_desc, iteration,
                        parent_code=parent_code, child_code=child_code, code_diff=code_diff,
                        parent_metrics=parent.metrics, child_metrics=child_metrics
                    )

    def _track_successes(
        self, iteration: int, result: Any, parent_metrics: Optional[Dict[str, float]]
    ) -> None:
        """Track success patterns from iteration result"""
        child_metrics = getattr(result, "child_metrics", None)
        child_program = getattr(result, "child_program", None)

        if not child_metrics or not child_program or not parent_metrics:
            return

        # Calculate improvement
        improvement = self._calculate_improvement(parent_metrics, child_metrics)

        if improvement >= self.config.min_improvement_threshold:
            # Extract what changed
            changes = getattr(child_program, "metadata", {}).get("changes", "Unknown")
            if changes and changes != "Unknown":
                # Extract code context
                parent = getattr(result, "parent", None)
                child_code = getattr(child_program, "code", None)
                parent_code = getattr(parent, "code", None) if parent else None
                code_diff = changes  # changes is already the diff summary

                self._add_success_pattern(
                    changes, iteration, improvement,
                    parent_code=parent_code, child_code=child_code, code_diff=code_diff,
                    parent_metrics=parent_metrics, child_metrics=child_metrics
                )

    def _extract_syntax_errors(self, artifacts: Dict[str, Any]) -> List[str]:
        """Extract syntax errors from artifacts"""
        errors = []

        # Check stderr for syntax errors
        stderr = artifacts.get("stderr", "")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")

        # Common Python syntax error patterns
        syntax_patterns = [
            (r"SyntaxError: (.+)", lambda m: f"SyntaxError: {m.group(1)}"),
            (r"IndentationError: (.+)", lambda m: f"IndentationError: {m.group(1)}"),
            (r"NameError: name ['\"](\w+)['\"] is not defined", lambda m: f"Undefined variable: {m.group(1)}"),
            (r"invalid syntax", lambda m: "Invalid syntax"),
        ]

        for pattern, formatter in syntax_patterns:
            matches = re.finditer(pattern, stderr)
            for match in matches:
                errors.append(formatter(match))

        return errors

    def _extract_runtime_errors(self, artifacts: Dict[str, Any]) -> List[str]:
        """Extract runtime errors from artifacts"""
        errors = []

        stderr = artifacts.get("stderr", "")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")

        # Common runtime error patterns
        runtime_patterns = [
            (r"IndexError: (.+)", lambda m: f"IndexError: {m.group(1)}"),
            (r"KeyError: (.+)", lambda m: f"KeyError: {m.group(1)}"),
            (r"ValueError: (.+)", lambda m: f"ValueError: {m.group(1)}"),
            (r"TypeError: (.+)", lambda m: f"TypeError: {m.group(1)}"),
            (r"AttributeError: (.+)", lambda m: f"AttributeError: {m.group(1)}"),
            (r"ZeroDivisionError", lambda m: "Division by zero"),
        ]

        for pattern, formatter in runtime_patterns:
            matches = re.finditer(pattern, stderr)
            for match in matches:
                errors.append(formatter(match))

        return errors

    def _detect_performance_regressions(
        self, parent_metrics: Dict[str, float], child_metrics: Dict[str, float]
    ) -> List[str]:
        """Detect performance regressions"""
        regressions = []

        for metric_name, child_value in child_metrics.items():
            if metric_name not in parent_metrics:
                continue

            parent_value = parent_metrics[metric_name]

            # Only compare numeric values
            if not isinstance(child_value, (int, float)) or not isinstance(
                parent_value, (int, float)
            ):
                continue

            # Check for significant regression using configurable threshold
            threshold_multiplier = 1.0 - self.config.performance_regression_threshold
            if parent_value > 0 and child_value < parent_value * threshold_multiplier:
                regression_pct = ((parent_value - child_value) / parent_value) * 100
                regressions.append(
                    f"{metric_name} decreased by {regression_pct:.1f}% "
                    f"({parent_value:.3f} → {child_value:.3f})"
                )

        return regressions

    def _calculate_improvement(
        self, parent_metrics: Dict[str, float], child_metrics: Dict[str, float]
    ) -> float:
        """Calculate overall improvement score"""
        # Prefer combined_score if available (consistent with rest of codebase)
        if "combined_score" in parent_metrics and "combined_score" in child_metrics:
            parent_score = parent_metrics["combined_score"]
            child_score = child_metrics["combined_score"]
            if isinstance(parent_score, (int, float)) and isinstance(child_score, (int, float)):
                if parent_score > 0:
                    return (child_score - parent_score) / parent_score

        # Fallback to averaging all metrics
        improvements = []
        for metric_name, child_value in child_metrics.items():
            if metric_name not in parent_metrics:
                continue

            parent_value = parent_metrics[metric_name]

            if not isinstance(child_value, (int, float)) or not isinstance(
                parent_value, (int, float)
            ):
                continue

            if parent_value > 0:
                improvement = (child_value - parent_value) / parent_value
                improvements.append(improvement)

        if improvements:
            return sum(improvements) / len(improvements)
        return 0.0

    def _normalize_error_description(self, description: str) -> str:
        """Normalize error descriptions for better grouping"""
        # Normalize container types (list/tuple/str → sequence)
        description = re.sub(r'\b(list|tuple|str|dict)\b index', 'sequence index', description)
        # Normalize numeric values to 'N'
        description = re.sub(r'\b\d+\b', 'N', description)
        # Lowercase for consistency
        return description.lower()

    def _summarize_pattern_with_llm_sync(
        self, pattern: Union[FailurePattern, SuccessPattern], is_failure: bool = True
    ) -> str:
        """
        Use LLM to generate actionable guidance from a pattern with code context (synchronous)

        Args:
            pattern: The pattern to summarize
            is_failure: Whether this is a failure pattern (vs success pattern)

        Returns:
            Natural language summary providing clear, actionable guidance
        """
        import asyncio

        logger.debug(f"_summarize_pattern_with_llm_sync called for {'failure' if is_failure else 'success'}")

        # Check if summarization is available
        if not self.llm_ensemble:
            logger.debug("No LLM ensemble available, returning raw description")
            return pattern.description

        # Check cache first
        cache_key = f"{'fail' if is_failure else 'success'}:{pattern.description}:{pattern.count}"
        if cache_key in self.summary_cache:
            logger.debug(f"Using cached summary for pattern")
            return self.summary_cache[cache_key]

        logger.debug(f"No cache hit, will call LLM to generate summary")

        # Build the same prompt as async version
        if is_failure:
            pattern_type = pattern.pattern_type
            prompt = f"""You are analyzing code evolution patterns to help guide future changes.

Pattern Type: {pattern_type}
Raw Description: {pattern.description}
Occurrences: {pattern.count} times (iterations {pattern.first_seen}-{pattern.last_seen})"""
        else:
            prompt = f"""You are analyzing code evolution patterns to help guide future changes.

Raw Description: {pattern.description}
Occurrences: {pattern.count} times
Average Improvement: +{pattern.avg_improvement:.1%}"""

        # Add code context if available
        if pattern.code_diff and pattern.code_diff != "Full rewrite":
            prompt += f"\n\nCode Changes:\n{pattern.code_diff}"
        elif pattern.code_diff == "Full rewrite" and pattern.parent_code and pattern.child_code:
            # For full rewrites, include both parent and child code
            prompt += f"\n\nParent Code:\n{pattern.parent_code}\n\nChild Code:\n{pattern.child_code}"
        else:
            # No code context available - log warning
            logger.warning(f"No code context available for pattern: {pattern.description[:100]}... (code_diff={pattern.code_diff}, has_parent={bool(pattern.parent_code)}, has_child={bool(pattern.child_code)})")

        # Add metrics context if available
        if pattern.parent_metrics and pattern.child_metrics:
            metrics_str = "\n\nMetrics Change:"
            metric_changes = []
            for key in pattern.child_metrics:
                if key in pattern.parent_metrics:
                    parent_val = pattern.parent_metrics[key]
                    child_val = pattern.child_metrics[key]
                    if isinstance(parent_val, (int, float)) and isinstance(child_val, (int, float)):
                        if parent_val != 0:
                            change_pct = ((child_val - parent_val) / parent_val) * 100
                            metric_changes.append((key, parent_val, child_val, abs(change_pct)))

            metric_changes.sort(key=lambda x: x[3], reverse=True)
            for key, parent_val, child_val, _ in metric_changes[:3]:
                metrics_str += f"\n- {key}: {parent_val:.4f} → {child_val:.4f}"

            if metric_changes:
                prompt += metrics_str

        if is_failure:
            prompt += "\n\nGenerate a 2-sentence summary:\n1. First sentence: Describe what code change or pattern caused this failure (be specific about what happened in the code).\n2. Second sentence: State what to avoid or watch out for when making similar changes in the future."
        else:
            prompt += "\n\nGenerate a 2-sentence summary:\n1. First sentence: Describe what code change or pattern led to this improvement (be specific about what happened in the code).\n2. Second sentence: State what to apply or consider when making similar changes in the future."

        try:
            # Call LLM - need to handle running event loop
            logger.debug(f"Calling llm_ensemble.generate_with_context with prompt of length {len(prompt)}")

            # Helper function to run the async call
            async def _call_llm():
                return await self.llm_ensemble.generate_with_context(
                    system_message="You are an expert code evolution advisor. Provide concise, actionable guidance.",
                    messages=[{"role": "user", "content": prompt}]
                )

            # Try to detect if there's a running event loop
            try:
                loop = asyncio.get_running_loop()
                # Event loop is running, use thread pool
                logger.debug("Event loop detected, using thread pool for LLM call")
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    summary = executor.submit(asyncio.run, _call_llm()).result()
            except RuntimeError:
                # No running event loop, use asyncio.run directly
                logger.debug("No event loop running, using asyncio.run directly")
                summary = asyncio.run(_call_llm())

            logger.debug(f"LLM call successful, received summary: {summary[:200]}...")

            # Clean up the summary
            summary = summary.strip().strip('"').strip()

            # Cache it
            self.summary_cache[cache_key] = summary
            logger.debug(f"Cached summary with key: {cache_key[:50]}...")

            return summary
        except Exception as e:
            logger.warning(f"Failed to generate LLM summary: {e}")
            return pattern.description

    async def _summarize_pattern_with_llm(
        self, pattern: Union[FailurePattern, SuccessPattern], is_failure: bool = True
    ) -> str:
        """
        Use LLM to generate actionable guidance from a pattern with code context

        Args:
            pattern: The pattern to summarize
            is_failure: Whether this is a failure pattern (vs success pattern)

        Returns:
            Natural language summary providing clear, actionable guidance
        """
        logger.debug(f"_summarize_pattern_with_llm called for {'failure' if is_failure else 'success'}")

        # Check if summarization is available
        if not self.llm_ensemble:
            logger.debug("No LLM ensemble available, returning raw description")
            # Fall back to raw description
            return pattern.description

        # Check cache first
        cache_key = f"{'fail' if is_failure else 'success'}:{pattern.description}:{pattern.count}"
        if cache_key in self.summary_cache:
            logger.debug(f"Using cached summary for pattern")
            return self.summary_cache[cache_key]

        logger.debug(f"No cache hit, will call LLM to generate summary")

        # Build prompt for LLM
        if is_failure:
            pattern_type = pattern.pattern_type
            prompt = f"""You are helping guide code evolution. Convert this failure pattern into clear, actionable advice.

Pattern Type: {pattern_type}
Description: {pattern.description}
Occurrences: {pattern.count} times (iterations {pattern.first_seen}-{pattern.last_seen})"""
        else:
            prompt = f"""You are helping guide code evolution. Convert this success pattern into clear, actionable advice.

Description: {pattern.description}
Occurrences: {pattern.count} times
Average Improvement: +{pattern.avg_improvement:.1%}"""

        # Add code context if available
        if pattern.code_diff and pattern.code_diff != "Full rewrite":
            prompt += f"\n\nCode Changes:\n{pattern.code_diff}"
        elif pattern.code_diff == "Full rewrite" and pattern.parent_code and pattern.child_code:
            # For full rewrites, include both parent and child code
            prompt += f"\n\nParent Code:\n{pattern.parent_code}\n\nChild Code:\n{pattern.child_code}"
        else:
            # No code context available - log warning
            logger.warning(f"No code context available for pattern: {pattern.description[:100]}... (code_diff={pattern.code_diff}, has_parent={bool(pattern.parent_code)}, has_child={bool(pattern.child_code)})")

        # Add metrics context if available
        if pattern.parent_metrics and pattern.child_metrics:
            metrics_str = "\n\nMetrics Change:"
            # Show top 3 most significant metric changes
            metric_changes = []
            for key in pattern.child_metrics:
                if key in pattern.parent_metrics:
                    parent_val = pattern.parent_metrics[key]
                    child_val = pattern.child_metrics[key]
                    if isinstance(parent_val, (int, float)) and isinstance(child_val, (int, float)):
                        if parent_val != 0:
                            change_pct = ((child_val - parent_val) / parent_val) * 100
                            metric_changes.append((key, parent_val, child_val, abs(change_pct)))

            # Sort by absolute change and take top 3
            metric_changes.sort(key=lambda x: x[3], reverse=True)
            for key, parent_val, child_val, _ in metric_changes[:3]:
                metrics_str += f"\n- {key}: {parent_val:.4f} → {child_val:.4f}"

            if metric_changes:
                prompt += metrics_str

        prompt += "\n\nGenerate 1-2 sentences of clear, actionable guidance that will help avoid this issue (for failures) or encourage this pattern (for successes). Be specific and practical."

        try:
            # Call LLM
            logger.debug(f"Calling llm_ensemble.generate_with_context with prompt of length {len(prompt)}")
            summary = await self.llm_ensemble.generate_with_context(
                system_message="You are an expert code evolution advisor. Provide concise, actionable guidance.",
                messages=[{"role": "user", "content": prompt}]
            )
            logger.debug(f"LLM call successful, received summary: {summary[:200]}...")

            # Clean up the summary (remove quotes, extra whitespace)
            summary = summary.strip().strip('"').strip()

            # Cache it
            self.summary_cache[cache_key] = summary
            logger.debug(f"Cached summary with key: {cache_key[:50]}...")

            return summary
        except Exception as e:
            logger.warning(f"Failed to generate LLM summary: {e}")
            # Fall back to raw description
            return pattern.description

    def _add_failure_pattern(
        self,
        pattern_type: str,
        description: str,
        iteration: int,
        example_error: Optional[str] = None,
        parent_code: Optional[str] = None,
        child_code: Optional[str] = None,
        code_diff: Optional[str] = None,
        parent_metrics: Optional[Dict[str, float]] = None,
        child_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Add or update a failure pattern"""
        # Normalize description for better grouping
        normalized_desc = self._normalize_error_description(description)
        key = f"{pattern_type}:{normalized_desc}"

        if key in self.failure_patterns:
            pattern = self.failure_patterns[key]
            pattern.count += 1
            pattern.last_seen = iteration
            logger.debug(f"Updated failure pattern: {description} (count={pattern.count})")
        else:
            self.failure_patterns[key] = FailurePattern(
                pattern_type=pattern_type,
                description=description,
                count=1,
                first_seen=iteration,
                last_seen=iteration,
                example_error=example_error,
                parent_code=parent_code,
                child_code=child_code,
                code_diff=code_diff,
                parent_metrics=parent_metrics,
                child_metrics=child_metrics,
            )
            logger.info(f"New failure pattern detected: {description} ({pattern_type})")

    def _add_success_pattern(
        self,
        description: str,
        iteration: int,
        improvement: float,
        parent_code: Optional[str] = None,
        child_code: Optional[str] = None,
        code_diff: Optional[str] = None,
        parent_metrics: Optional[Dict[str, float]] = None,
        child_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Add or update a success pattern"""
        key = description

        if key in self.success_patterns:
            pattern = self.success_patterns[key]
            # Update average improvement
            total_improvement = pattern.avg_improvement * pattern.count + improvement
            pattern.count += 1
            pattern.avg_improvement = total_improvement / pattern.count
            pattern.last_seen = iteration
            logger.debug(
                f"Updated success pattern: {description} "
                f"(count={pattern.count}, avg_improvement={pattern.avg_improvement:.2%})"
            )
        else:
            self.success_patterns[key] = SuccessPattern(
                description=description,
                count=1,
                avg_improvement=improvement,
                first_seen=iteration,
                last_seen=iteration,
                parent_code=parent_code,
                child_code=child_code,
                code_diff=code_diff,
                parent_metrics=parent_metrics,
                child_metrics=child_metrics,
            )
            logger.info(
                f"New success pattern detected: {description} (+{improvement:.2%})"
            )

    def get_top_failures(self, max_count: Optional[int] = None) -> List[FailurePattern]:
        """Get top failure patterns sorted by count"""
        max_count = max_count or self.config.max_learnings

        # Filter by minimum count
        filtered = [
            p for p in self.failure_patterns.values() if p.count >= self.config.min_failure_count
        ]

        # Sort by count (descending)
        sorted_patterns = sorted(filtered, key=lambda p: p.count, reverse=True)

        return sorted_patterns[:max_count]

    def get_top_successes(self, max_count: Optional[int] = None) -> List[SuccessPattern]:
        """Get top success patterns sorted by count and improvement"""
        max_count = max_count or self.config.max_learnings

        # Filter by minimum count
        filtered = [
            p for p in self.success_patterns.values() if p.count >= self.config.min_success_count
        ]

        # Sort by count * avg_improvement (descending)
        sorted_patterns = sorted(
            filtered, key=lambda p: p.count * p.avg_improvement, reverse=True
        )

        return sorted_patterns[:max_count]

    def generate_prompt_section(self) -> str:
        """
        Generate formatted section for prompt injection (with LLM summarization)

        Returns:
            Formatted string with learnings, or empty string if disabled or no learnings
        """
        if not self.config.enabled:
            return ""

        sections = []

        # Add failures section
        if self.config.track_failures or self.config.track_both:
            failures = self.get_top_failures()
            if failures:
                sections.append(self._format_failures_section_sync(failures))

        # Add successes section
        if self.config.track_successes or self.config.track_both:
            successes = self.get_top_successes()
            if successes:
                sections.append(self._format_successes_section_sync(successes))

        if not sections:
            return ""

        header = "## Evolution Insights (Global Learnings)"
        if self.config.verbosity == "minimal":
            header = "## Common Patterns"

        return f"{header}\n\n" + "\n\n".join(sections)

    def _generate_prompt_section_sync(self) -> str:
        """Synchronous version without LLM summarization"""
        sections = []

        # Add failures section
        if self.config.track_failures or self.config.track_both:
            failures = self.get_top_failures()
            if failures:
                sections.append(self._format_failures_section(failures))

        # Add successes section
        if self.config.track_successes or self.config.track_both:
            successes = self.get_top_successes()
            if successes:
                sections.append(self._format_successes_section(successes))

        if not sections:
            return ""

        header = "## Evolution Insights (Global Learnings)"
        if self.config.verbosity == "minimal":
            header = "## Common Patterns"

        return f"{header}\n\n" + "\n\n".join(sections)

    async def _generate_prompt_section_async(self) -> str:
        """Async version with LLM summarization"""
        sections = []

        # Add failures section
        if self.config.track_failures or self.config.track_both:
            failures = self.get_top_failures()
            if failures:
                section = await self._format_failures_section_async(failures)
                sections.append(section)

        # Add successes section
        if self.config.track_successes or self.config.track_both:
            successes = self.get_top_successes()
            if successes:
                section = await self._format_successes_section_async(successes)
                sections.append(section)

        if not sections:
            return ""

        header = "## Evolution Insights (Global Learnings)"
        if self.config.verbosity == "minimal":
            header = "## Common Patterns"

        return f"{header}\n\n" + "\n\n".join(sections)

    def _format_failures_section_sync(self, failures: List[FailurePattern]) -> str:
        """Format failures section with synchronous LLM summarization"""
        logger.debug(f"_format_failures_section_sync called with {len(failures)} failures")
        logger.debug(f"use_llm_summarization={self.config.use_llm_summarization}, llm_ensemble={'available' if self.llm_ensemble else 'None'}")
        lines = []

        if self.config.verbosity == "minimal":
            lines.append("### Avoid:")
            for f in failures:
                if self.config.use_llm_summarization and self.llm_ensemble:
                    logger.debug(f"Calling LLM to summarize failure pattern: {f.description[:50]}...")
                    summary = self._summarize_pattern_with_llm_sync(f, is_failure=True)
                    logger.debug(f"LLM summary received: {summary[:100]}...")
                    lines.append(f"- {summary} (seen {f.count}x)")
                else:
                    logger.debug(f"Skipping LLM, using raw description for: {f.description[:50]}...")
                    lines.append(f"- {f.description} (seen {f.count}x)")
        elif self.config.verbosity == "concise":
            lines.append("### Common Pitfalls:")
            for f in failures:
                icon = "❌" if f.pattern_type == "syntax" else "⚠️"
                if self.config.use_llm_summarization and self.llm_ensemble:
                    logger.debug(f"Calling LLM to summarize failure pattern: {f.description[:50]}...")
                    summary = self._summarize_pattern_with_llm_sync(f, is_failure=True)
                    logger.debug(f"LLM summary received: {summary[:100]}...")
                    lines.append(f"{icon} (seen {f.count}x) {summary}")
                else:
                    logger.debug(f"Skipping LLM, using raw description for: {f.description[:50]}...")
                    lines.append(f"{icon} (seen {f.count}x) {f.description}")
        else:  # detailed
            lines.append("### Common Pitfalls (from recent evolution):")
            for f in failures:
                icon = "❌" if f.pattern_type == "syntax" else "⚠️"
                if self.config.use_llm_summarization and self.llm_ensemble:
                    logger.debug(f"Calling LLM to summarize failure pattern: {f.description[:50]}...")
                    summary = self._summarize_pattern_with_llm_sync(f, is_failure=True)
                    logger.debug(f"LLM summary received: {summary[:100]}...")
                    lines.append(
                        f"{icon} **{f.pattern_type.replace('_', ' ').title()}**: "
                        f"{summary} (seen {f.count}x, last at iteration {f.last_seen})"
                    )
                else:
                    logger.debug(f"Skipping LLM, using raw description for: {f.description[:50]}...")
                    lines.append(
                        f"{icon} **{f.pattern_type.replace('_', ' ').title()}**: "
                        f"{f.description} (seen {f.count}x, last at iteration {f.last_seen})"
                    )

        return "\n".join(lines)

    def _format_successes_section_sync(self, successes: List[SuccessPattern]) -> str:
        """Format successes section with synchronous LLM summarization"""
        lines = []

        if self.config.verbosity == "minimal":
            lines.append("### Successful patterns:")
            for s in successes:
                if self.config.use_llm_summarization and self.llm_ensemble:
                    summary = self._summarize_pattern_with_llm_sync(s, is_failure=False)
                    lines.append(f"- {summary} (seen {s.count}x)")
                else:
                    lines.append(f"- {s.description} (seen {s.count}x)")
        elif self.config.verbosity == "concise":
            lines.append("### Successful Patterns:")
            for s in successes:
                if self.config.use_llm_summarization and self.llm_ensemble:
                    summary = self._summarize_pattern_with_llm_sync(s, is_failure=False)
                    lines.append(
                        f"✅ {summary} (seen {s.count}x, avg improvement: +{s.avg_improvement:.2%})"
                    )
                else:
                    lines.append(
                        f"✅ {s.description} (seen {s.count}x, avg improvement: +{s.avg_improvement:.2%})"
                    )
        else:  # detailed
            lines.append("### Successful Patterns (from recent evolution):")
            for s in successes:
                if self.config.use_llm_summarization and self.llm_ensemble:
                    summary = self._summarize_pattern_with_llm_sync(s, is_failure=False)
                    lines.append(
                        f"✅ **Success**: {summary} (seen {s.count}x, "
                        f"avg improvement: +{s.avg_improvement:.2%}, last at iteration {s.last_seen})"
                    )
                else:
                    lines.append(
                        f"✅ **Success**: {s.description} (seen {s.count}x, "
                        f"avg improvement: +{s.avg_improvement:.2%}, last at iteration {s.last_seen})"
                    )

        return "\n".join(lines)

    async def _format_failures_section_async(self, failures: List[FailurePattern]) -> str:
        """Format failures section with LLM summarization"""
        logger.debug(f"_format_failures_section_async called with {len(failures)} failures")
        logger.debug(f"use_llm_summarization={self.config.use_llm_summarization}, llm_ensemble={'available' if self.llm_ensemble else 'None'}")
        lines = []

        if self.config.verbosity == "minimal":
            lines.append("### Avoid:")
            for f in failures:
                if self.config.use_llm_summarization and self.llm_ensemble:
                    logger.debug(f"Calling LLM to summarize failure pattern: {f.description[:50]}...")
                    summary = await self._summarize_pattern_with_llm(f, is_failure=True)
                    logger.debug(f"LLM summary received: {summary[:100]}...")
                    lines.append(f"- {summary} (seen {f.count}x)")
                else:
                    logger.debug(f"Skipping LLM, using raw description for: {f.description[:50]}...")
                    lines.append(f"- {f.description} (seen {f.count}x)")
        elif self.config.verbosity == "concise":
            lines.append("### Common Pitfalls:")
            for f in failures:
                icon = "❌" if f.pattern_type == "syntax" else "⚠️"
                if self.config.use_llm_summarization and self.llm_ensemble:
                    logger.debug(f"Calling LLM to summarize failure pattern: {f.description[:50]}...")
                    summary = await self._summarize_pattern_with_llm(f, is_failure=True)
                    logger.debug(f"LLM summary received: {summary[:100]}...")
                    lines.append(f"{icon} {summary} (seen {f.count}x)")
                else:
                    logger.debug(f"Skipping LLM, using raw description for: {f.description[:50]}...")
                    lines.append(f"{icon} {f.description} (seen {f.count}x)")
        else:  # detailed
            lines.append("### Common Pitfalls (from recent evolution):")
            for f in failures:
                icon = "❌" if f.pattern_type == "syntax" else "⚠️"
                if self.config.use_llm_summarization and self.llm_ensemble:
                    logger.debug(f"Calling LLM to summarize failure pattern: {f.description[:50]}...")
                    summary = await self._summarize_pattern_with_llm(f, is_failure=True)
                    logger.debug(f"LLM summary received: {summary[:100]}...")
                    lines.append(
                        f"{icon} **{f.pattern_type.replace('_', ' ').title()}**: "
                        f"{summary} (seen {f.count}x, last at iteration {f.last_seen})"
                    )
                else:
                    logger.debug(f"Skipping LLM, using raw description for: {f.description[:50]}...")
                    lines.append(
                        f"{icon} **{f.pattern_type.replace('_', ' ').title()}**: "
                        f"{f.description} (seen {f.count}x, last at iteration {f.last_seen})"
                    )

        return "\n".join(lines)

    async def _format_successes_section_async(self, successes: List[SuccessPattern]) -> str:
        """Format successes section with LLM summarization"""
        lines = []

        if self.config.verbosity == "minimal":
            lines.append("### Successful patterns:")
            for s in successes:
                if self.config.use_llm_summarization and self.llm_ensemble:
                    summary = await self._summarize_pattern_with_llm(s, is_failure=False)
                    lines.append(f"- {summary} (seen {s.count}x)")
                else:
                    lines.append(f"- {s.description} (seen {s.count}x)")
        elif self.config.verbosity == "concise":
            lines.append("### Successful Patterns:")
            for s in successes:
                if self.config.use_llm_summarization and self.llm_ensemble:
                    summary = await self._summarize_pattern_with_llm(s, is_failure=False)
                    lines.append(
                        f"✅ {summary} (seen {s.count}x, avg improvement: +{s.avg_improvement:.2%})"
                    )
                else:
                    lines.append(
                        f"✅ {s.description} (seen {s.count}x, avg improvement: +{s.avg_improvement:.2%})"
                    )
        else:  # detailed
            lines.append("### Successful Patterns (from recent evolution):")
            for s in successes:
                if self.config.use_llm_summarization and self.llm_ensemble:
                    summary = await self._summarize_pattern_with_llm(s, is_failure=False)
                    lines.append(
                        f"✅ **Success**: {summary} (seen {s.count}x, "
                        f"avg improvement: +{s.avg_improvement:.2%}, last at iteration {s.last_seen})"
                    )
                else:
                    lines.append(
                        f"✅ **Success**: {s.description} (seen {s.count}x, "
                        f"avg improvement: +{s.avg_improvement:.2%}, last at iteration {s.last_seen})"
                    )

        return "\n".join(lines)

    def save(self, checkpoint_dir: Path) -> None:
        """
        Save global learnings to checkpoint directory

        Args:
            checkpoint_dir: Directory to save checkpoint data
        """
        if not self.config.enabled:
            return

        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "failure_patterns": {k: v.to_dict() for k, v in self.failure_patterns.items()},
            "success_patterns": {k: v.to_dict() for k, v in self.success_patterns.items()},
            "iteration_history": self.iteration_history,
            "last_update_iteration": self.last_update_iteration,
        }

        save_path = checkpoint_dir / "global_learnings.json"
        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved global learnings to {save_path}")

    def load(self, checkpoint_dir: Path) -> None:
        """
        Load global learnings from checkpoint directory

        Args:
            checkpoint_dir: Directory containing checkpoint data
        """
        if not self.config.enabled:
            return

        checkpoint_dir = Path(checkpoint_dir)
        load_path = checkpoint_dir / "global_learnings.json"

        if not load_path.exists():
            logger.warning(f"Global learnings checkpoint not found at {load_path}")
            return

        with open(load_path, "r") as f:
            data = json.load(f)

        # Restore failure patterns
        self.failure_patterns = {
            k: FailurePattern.from_dict(v) for k, v in data.get("failure_patterns", {}).items()
        }

        # Restore success patterns
        self.success_patterns = {
            k: SuccessPattern.from_dict(v) for k, v in data.get("success_patterns", {}).items()
        }

        # Restore metadata
        self.iteration_history = data.get("iteration_history", [])
        self.last_update_iteration = data.get("last_update_iteration", 0)

        logger.info(
            f"Loaded global learnings from {load_path} "
            f"({len(self.failure_patterns)} failures, {len(self.success_patterns)} successes)"
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for logging"""
        return {
            "total_failures": len(self.failure_patterns),
            "total_successes": len(self.success_patterns),
            "iterations_tracked": len(self.iteration_history),
            "last_update": self.last_update_iteration,
            "top_failures": len(self.get_top_failures()),
            "top_successes": len(self.get_top_successes()),
        }
