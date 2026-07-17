# Experiment matrix

The primary comparison changes the checkpoint while holding fixed the
full-distribution sampling law, identical raw WikiText prompt text, seed
scheme, and continuation checkpoints. Each checkpoint stays in its native
stored precision; model architecture, native precision, and required backend
are therefore inseparable parts of the model condition. Prompt and
chat-template tokens are conditioning context, not observations.

## Model identities and feasibility

| Requested family | Exact checkpoint | Native context | Local status |
|---|---|---:|---|
| Gemma 3 4B IT | `google/gemma-3-4b-it` | 128K | Gated; accept the Gemma license and authenticate first |
| Qwen 3 30B A3B Instruct | `Qwen/Qwen3-30B-A3B-Instruct-2507` | 262K | Native BF16 weights need substantially more than this machine's 16 GB |
| Qwen 3 Next A3B Instruct | `Qwen/Qwen3-Next-80B-A3B-Instruct` | 262K | This is an 80B model; there is no 30B Qwen3-Next A3B checkpoint |
| Gemma 1 2B IT | `google/gemma-2b-it` | 8K | Gated; literal `Gemma-2B-it` checkpoint in this matrix |
| Llama 2 7B | `meta-llama/Llama-2-7b-hf` | 4K | Gated; base rather than chat checkpoint |
| Mamba 130M | `state-spaces/mamba-130m-hf` | recurrent | Public; feasible local smoke-test model |

The phrase “Qwen-Next-30B-A3B-Instruct” combines two different public model
families. We use the 30B Qwen3 checkpoint for the requested 30B experiment and
list the actual Qwen3-Next model separately.

Quantized weights are intentionally not substituted for either Qwen model:
quantization changes the sampling distribution, so it would define a different
experiment. A quantized run may be added later as an explicitly separate model
condition.

The matrix uses `Qwen/Qwen3-30B-A3B-Instruct-2507`, not Qwen3-Next. It uses
`google/gemma-2b-it` for the requested Gemma-2B-it; Gemma 2 2B IT would be the
distinct checkpoint `google/gemma-2-2b-it`.

## Planned matrix

| condition | prompts | rollouts | continuation checkpoints |
|---|---:|---:|---|
| unconditional | none; one unevaluated seed | 32 per model | powers of two, 32–2,048 |
| WikiText | 100 distinct article excerpts | 64 per prompt/model | powers of two, 32–2,048 |

Instruction-tuned models sampled unconditionally are intentionally off their
chat template. That is a legitimate self-sampling condition, but it is not
normal assistant behavior. Native-chat sampling is a separate sensitivity
condition and must not be pooled into the raw-prefix comparison.

The primary prompt file is
`prompts/wikitext-2-train-articles-mamba-1024-100.jsonl`: 100 articles sampled
without replacement with seed 0 from the 567 of 629 pinned WikiText-2 raw
training articles long enough to provide 1,024 Mamba-tokenizer tokens. Each
prompt is the article prefix, so this is a long-article-prefix population rather
than an unbiased sample of all WikiText articles. Every model receives identical
text; runtime metadata records its model-specific prompt length.

Preview exact commands:

```bash
.venv/bin/canonicity-matrix \
  --all-models \
  --condition unconditional \
  --condition wikitext \
  --dry-run
```

Submit large jobs independently:

```bash
.venv/bin/canonicity-matrix \
  --model qwen3-30b-a3b-instruct-2507 \
  --condition wikitext \
  --output-root results/model-matrix
```

Each completed generation batch is atomically stored under the job directory.
Repeating an interrupted command resumes only when the resolved commits,
hardware placement, prompts, and all sampling settings match exactly. Dense
prefix analysis is CPU-parallel (matrix default: four workers) and can be
restarted independently from `samples.jsonl`.

Logs are unbuffered and phase-specific. The matrix wrapper first prints job
`N/M` and its planned contexts and rollouts. After loading, the child reports
actual model placement (`GPU (CUDA)`, `CPU`, or mixed/offloaded), parameter
footprint by device, checkpointed rollouts, rollouts still requiring model
sampling, and unfinished contexts. Evaluation and segment-analysis logs also
include completed and remaining rollout counts. Sampling counts move only at a
completed batch boundary; reduce `--batch-size` if a single generation call is
too long to provide useful progress granularity.

Only one process may write a given job directory. Contexts are all tokenized
and checked against the model context window before the first batch is stored.

All models default to `dtype=auto`, which preserves their checkpoint-native
precision. Qwen additionally defaults to Transformers/Accelerate
`device_map=auto`; other models default to one device. Pass `--device-map auto`
when sharding is needed. To prospectively pin commits, repeat
`--model-revision ALIAS=COMMIT`; resolved commits are always frozen into each
job's durable sampling plan before its first batch.

After every planned job finishes, correct the complete recurrence-test family:

```bash
.venv/bin/canonicity-aggregate \
  --results-root results/model-matrix
```

Aggregation validates the full 5-model × 2-condition × 5-horizon family,
sampling-plan compatibility, prompt identity, artifact hashes, and
recurrence-estimand identity. It recomputes recurrence from `segments.jsonl`
and applies Benjamini-Yekutieli correction in log space. `recurrence_all.csv`
includes ordinary and log10 p/q values, with untestable planned hypotheses
conservatively assigned multiplicity `p=1`;
`canonicity_all.csv` combines the primary canonicity curves. Incomplete matrices
are rejected rather than quietly redefining the family.

## Authentication

Gemma and Llama 2 require accepting their respective Hugging Face licenses in a
browser and then authenticating:

```bash
.venv/bin/hf auth login
.venv/bin/hf auth whoami
```

## Unconditional long continuations

Use the same command shape for every model:

```bash
.venv/bin/canonicity \
  --model MODEL_ID \
  --samples 64 \
  --lengths 1:128,256:2048:128 \
  --batch-size 1 \
  --seed 0 \
  --dtype auto \
  --output OUTPUT_DIRECTORY
```

For Mamba-130M on CPU, batch size 32 is appropriate. `dtype=auto` keeps Mamba in
FP32, Llama 2 in native FP16, and Gemma/Qwen in native BF16. Do not compare a
native-precision model with a silently cast or quantized substitute.

## Dataset-conditioned long continuations (small exploratory recipe)

The checked-in prompt file contains four sequential raw WikiText contexts,
each exactly 1,024 tokens under the Mamba tokenizer. Reuse its text unchanged
for paired cross-model comparison; `metadata.json` records each tested model's
actual prompt length.

```bash
.venv/bin/canonicity \
  --model MODEL_ID \
  --prompts-file prompts/wikitext-2-test-mamba-1024.jsonl \
  --prompt-mode raw \
  --samples 16 \
  --lengths 1:128,256:1024:128 \
  --batch-size 1 \
  --seed 0 \
  --dtype auto \
  --output OUTPUT_DIRECTORY
```

The primary conditioned result is `pooled_summary.csv`, which pools the four
prompt-specific sequence counts. `summary.csv` retains each prompt separately
so context sensitivity is visible. Within-prompt Wilson 95% intervals make the
smaller long-run sample sizes explicit; the pooled percentage is descriptive
and has no iid-rollout interval.

Raw-prefix conditioning is the primary comparison because it preserves the
same continuation task across model families. `--prompt-mode chat` is a
separate sensitivity experiment for native instruction behavior; it should not
be mixed into the primary curve.
