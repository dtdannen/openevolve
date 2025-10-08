#!/usr/bin/env python
"""
Inspect global learnings from OpenEvolve checkpoints

Usage:
    python scripts/inspect_global_learnings.py <checkpoint_dir>
    python scripts/inspect_global_learnings.py examples/circle_packing/openevolve_output/checkpoints/checkpoint_50
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any


def load_global_learnings(checkpoint_path: Path) -> Dict[str, Any]:
    """Load global learnings JSON from checkpoint directory"""
    learnings_file = checkpoint_path / "global_learnings.json"

    if not learnings_file.exists():
        print(f"❌ No global_learnings.json found in {checkpoint_path}")
        print(f"   (Global learnings may be disabled)")
        return None

    with open(learnings_file, 'r') as f:
        return json.load(f)


def format_failure_pattern(key: str, pattern: Dict[str, Any]) -> str:
    """Format a failure pattern for display"""
    icon = "❌" if pattern['pattern_type'] == "syntax" else "⚠️"
    return (
        f"{icon} {pattern['description']}\n"
        f"   Type: {pattern['pattern_type']}, Count: {pattern['count']}, "
        f"First: iter {pattern['first_seen']}, Last: iter {pattern['last_seen']}"
    )


def format_success_pattern(key: str, pattern: Dict[str, Any]) -> str:
    """Format a success pattern for display"""
    return (
        f"✅ {pattern['description']}\n"
        f"   Count: {pattern['count']}, Avg improvement: +{pattern['avg_improvement']:.2%}, "
        f"First: iter {pattern['first_seen']}, Last: iter {pattern['last_seen']}"
    )


def print_learnings(data: Dict[str, Any]):
    """Pretty print global learnings data"""

    print("\n" + "="*80)
    print("GLOBAL LEARNINGS SUMMARY")
    print("="*80)

    # Metadata
    print(f"\n📊 Metadata:")
    print(f"   Iterations tracked: {len(data.get('iteration_history', []))}")
    print(f"   Last update: iteration {data.get('last_update_iteration', 0)}")
    print(f"   Window: iterations {min(data.get('iteration_history', [0]))} - "
          f"{max(data.get('iteration_history', [0]))}")

    # Failure patterns
    failure_patterns = data.get('failure_patterns', {})
    print(f"\n🔴 Failure Patterns ({len(failure_patterns)} total):")

    if failure_patterns:
        # Sort by count
        sorted_failures = sorted(
            failure_patterns.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )

        for i, (key, pattern) in enumerate(sorted_failures[:10], 1):  # Top 10
            print(f"\n{i}. {format_failure_pattern(key, pattern)}")

        if len(sorted_failures) > 10:
            print(f"\n   ... and {len(sorted_failures) - 10} more")
    else:
        print("   (No failure patterns detected yet)")

    # Success patterns
    success_patterns = data.get('success_patterns', {})
    print(f"\n🟢 Success Patterns ({len(success_patterns)} total):")

    if success_patterns:
        # Sort by count * avg_improvement
        sorted_successes = sorted(
            success_patterns.items(),
            key=lambda x: x[1]['count'] * x[1]['avg_improvement'],
            reverse=True
        )

        for i, (key, pattern) in enumerate(sorted_successes[:10], 1):  # Top 10
            print(f"\n{i}. {format_success_pattern(key, pattern)}")

        if len(sorted_successes) > 10:
            print(f"\n   ... and {len(sorted_successes) - 10} more")
    else:
        print("   (No success patterns detected yet)")

    print("\n" + "="*80)


def generate_prompt_preview(data: Dict[str, Any], max_learnings: int = 5,
                            min_failure_count: int = 3,
                            min_success_count: int = 3):
    """Generate what would be injected into prompts"""

    print("\n" + "="*80)
    print("PROMPT INJECTION PREVIEW")
    print("="*80)
    print("\nThis is what would be added to the system prompt:\n")
    print("-" * 80)

    # Filter and sort failures
    failures = data.get('failure_patterns', {})
    filtered_failures = {
        k: v for k, v in failures.items()
        if v['count'] >= min_failure_count
    }
    sorted_failures = sorted(
        filtered_failures.items(),
        key=lambda x: x[1]['count'],
        reverse=True
    )[:max_learnings]

    # Filter and sort successes
    successes = data.get('success_patterns', {})
    filtered_successes = {
        k: v for k, v in successes.items()
        if v['count'] >= min_success_count
    }
    sorted_successes = sorted(
        filtered_successes.items(),
        key=lambda x: x[1]['count'] * x[1]['avg_improvement'],
        reverse=True
    )[:max_learnings]

    # Generate prompt text
    if sorted_failures or sorted_successes:
        print("## Evolution Insights (Global Learnings)")
        print()

        if sorted_failures:
            print("### Common Pitfalls:")
            for _, pattern in sorted_failures:
                icon = "❌" if pattern['pattern_type'] == "syntax" else "⚠️"
                print(f"{icon} {pattern['description']} (seen {pattern['count']}x)")

        if sorted_successes:
            if sorted_failures:
                print()
            print("### Successful Patterns:")
            for _, pattern in sorted_successes:
                print(f"✅ {pattern['description']} (seen {pattern['count']}x, "
                      f"avg improvement: +{pattern['avg_improvement']:.2%})")
    else:
        print("(No learnings meet the threshold for injection)")

    print("-" * 80)
    print(f"\nThresholds: min_failure_count={min_failure_count}, "
          f"min_success_count={min_success_count}, max_learnings={max_learnings}")
    print("="*80)


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/inspect_global_learnings.py <checkpoint_dir>")
        print("\nExample:")
        print("  python scripts/inspect_global_learnings.py "
              "examples/circle_packing/openevolve_output/checkpoints/checkpoint_50")
        sys.exit(1)

    checkpoint_path = Path(sys.argv[1])

    if not checkpoint_path.exists():
        print(f"❌ Checkpoint directory not found: {checkpoint_path}")
        sys.exit(1)

    if not checkpoint_path.is_dir():
        print(f"❌ Not a directory: {checkpoint_path}")
        sys.exit(1)

    print(f"📂 Loading from: {checkpoint_path}")

    data = load_global_learnings(checkpoint_path)

    if data is None:
        sys.exit(1)

    print_learnings(data)
    generate_prompt_preview(data)

    print("\n✅ Inspection complete!\n")


if __name__ == "__main__":
    main()
