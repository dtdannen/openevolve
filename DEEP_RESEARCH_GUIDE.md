# Deep Research System Prompt Augmentation

This feature enables OpenEvolve to automatically enhance its system prompt during evolution using OpenAI's Deep Research API, combined with global learnings from the evolution process.

## Overview

Every N iterations (configurable), OpenEvolve will:
1. Collect global learnings from the evolution (best programs, failure patterns, statistics)
2. Query Deep Research for insights on improving code evolution strategies
3. Use an LLM to generate an augmented system prompt section
4. Append the new insights to the base prompt (never removes content)
5. Save versioned prompts to `{output_dir}/augmented_system_prompts/`

## Configuration

Add the `deep_research` section to your config YAML:

```yaml
deep_research:
  enabled: true
  augmentation_interval: 10  # Augment every N iterations
  model: "o3-deep-research-2025-06-26"  # or "o4-mini-deep-research-2025-06-26"
  timeout: 600  # Max seconds per research call
  max_tokens: 16000  # Max tokens in research output
```

### Configuration Options

- **enabled** (bool): Enable/disable Deep Research augmentation
- **augmentation_interval** (int): How often to augment (in iterations). Default: 10
- **model** (str): Deep Research model to use:
  - `o3-deep-research-2025-06-26`: Higher quality, slower
  - `o4-mini-deep-research-2025-06-26`: Faster, cheaper
- **timeout** (int): Maximum time to wait for research (seconds). Default: 600
- **max_tokens** (int): Maximum tokens in research output. Default: 16000

## File Structure

When Deep Research is enabled, augmented prompts are saved to:

```
{output_dir}/
└── augmented_system_prompts/
    ├── 0.txt       # Original base prompt
    ├── 10.txt      # After first augmentation (iteration 10)
    ├── 20.txt      # After second augmentation (iteration 20)
    └── ...
```

Each file contains the cumulative prompt (base + all augmentations so far).

## Example Usage

```bash
# Run with Deep Research enabled
python openevolve-run.py \
  examples/circle_packing/initial_program.py \
  examples/circle_packing/evaluator.py \
  --config examples/circle_packing/config_with_deep_research.yaml \
  --iterations 30
```

After iteration 10 and 20, you'll see:
```
🔬 Deep Research augmentation interval reached at iteration 10
Collected global learnings (stub)
Built research query (XXX chars)
Received Deep Research results (XXX chars)
Generated augmented prompt (XXX chars)
✅ System prompt augmented and saved to 10.txt
```

## How It Works

### 1. Global Learnings Collection (Stub)

Currently uses a simple stub that extracts:
- Total programs evaluated
- Best program score and iteration
- Number of islands
- Programming language

**Future**: Will integrate with full global learnings system to extract:
- Success patterns (what mutations worked)
- Failure patterns (common errors)
- Convergence statistics
- Diversity metrics

### 2. Deep Research Query

The system builds a research query using the template:
```
Research Query: How can we improve code evolution strategies for {language} programs?

Current Context:
- Iteration: {iteration} of evolution completed
- Global Learnings: {learnings_summary}

Base System Prompt Being Used:
{base_prompt}

Please research best practices, common pitfalls, and optimization strategies...
```

### 3. Meta-Prompting for Augmentation

Deep Research results are fed to an LLM (using your configured models) to generate a concise augmentation section:

```
## Deep Research Insights (Added at iteration {N}):
- Insight 1
- Insight 2
- ...
```

This is appended to the base prompt.

### 4. Prompt Versioning

Each augmented version is saved with the iteration number as filename, allowing you to:
- Track prompt evolution over time
- Analyze which insights were added when
- Resume from checkpoints with correct prompt version

## Cost Considerations

**Deep Research calls are expensive!** Consider:

- **Model choice**: `o4-mini-deep-research` is cheaper than `o3-deep-research`
- **Augmentation interval**: Every 10 iterations means 10 research calls for 100 iterations
- **Run length**: Shorter test runs = fewer research calls
- **Timeout**: Shorter timeouts may reduce costs but give less thorough research

Example costs (approximate):
- `o3-deep-research`: ~$X per call (depends on query complexity)
- `o4-mini-deep-research`: ~$Y per call (cheaper alternative)

For a 100-iteration run with interval=10:
- Research calls: 10 (at iterations 10, 20, 30, ..., 100)
- Estimated cost: 10 × $X = $Z

## Troubleshooting

### "Deep Research service unavailable"
- Check your OpenAI API key is valid
- Verify the Deep Research API is accessible
- Check timeout setting (may need to increase)
- Evolution will continue with current prompt if research fails

### Augmentation not happening
- Ensure `deep_research.enabled: true` in config
- Check iteration number is a multiple of `augmentation_interval`
- Look for error logs in `{output_dir}/logs/`

### Prompts not being saved
- Verify `output_dir` is writable
- Check `{output_dir}/augmented_system_prompts/` directory exists
- Look for permission errors in logs

## Advanced Usage

### Custom Templates

You can override the default templates in `openevolve/prompt/templates.py`:

- **deep_research_query**: Customize the research query format
- **augmentation_meta_prompt**: Customize how augmentations are generated

### Analyzing Prompt Evolution

```python
# Load and compare prompt versions
import os

prompts_dir = "examples/circle_packing/openevolve_output/augmented_system_prompts"

base_prompt = open(os.path.join(prompts_dir, "0.txt")).read()
v1_prompt = open(os.path.join(prompts_dir, "10.txt")).read()
v2_prompt = open(os.path.join(prompts_dir, "20.txt")).read()

# See what was added
print("=== Iteration 10 Addition ===")
print(v1_prompt[len(base_prompt):])

print("\\n=== Iteration 20 Addition ===")
print(v2_prompt[len(v1_prompt):])
```

## Future Enhancements

Planned improvements:
- [ ] Full global learnings integration (replace stub)
- [ ] Configurable research topics/focus areas
- [ ] Prompt compression when exceeding token limits
- [ ] A/B testing between augmented and non-augmented runs
- [ ] Visualization of prompt evolution impact on scores
