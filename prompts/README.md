# Dataset prompts

`wikitext-2-test-mamba-1024.jsonl` contains the original four sequential test
excerpts used for the exploratory Mamba run.

`wikitext-2-train-articles-mamba-1024-100.jsonl` is the confirmatory matrix
prompt set. Of 629 WikiText-2 training articles, 567 are long enough to provide
a 1,024-token prefix under the pinned Mamba tokenizer. It samples 100 of those
eligible articles without replacement with seed 0 and takes one prefix per
article. This prevents adjacent chunks from the same article being treated as
independent prompt strata, while making the long-article/prefix selection bias
explicit. Of the prompts, 99 contain exactly 1,024 Mamba-tokenizer tokens and
one contains 1,023; the latter backs off one token because detaching the source
prefix changes its final canonical merge boundary.

Both files were materialized from
`Salesforce/wikitext` commit `b08601e04326c79dfdd32d625aee71d232d685c3`
with `state-spaces/mamba-130m-hf` tokenizer commit
`1e76775f628fbf1350fbe4dbb3d971ba64af25a1`. Every record carries revision and
source-row provenance; the 100-article file additionally records document and
selection provenance.

Recreate the same document selection and prompt texts:

```bash
.venv/bin/canonicity-prompts \
  --dataset Salesforce/wikitext \
  --config wikitext-2-raw-v1 \
  --split train \
  --dataset-revision b08601e04326c79dfdd32d625aee71d232d685c3 \
  --tokenizer state-spaces/mamba-130m-hf \
  --tokenizer-revision 1e76775f628fbf1350fbe4dbb3d971ba64af25a1 \
  --prompt-tokens 1024 \
  --count 100 \
  --document-start-regex '^= [^=].* =$' \
  --selection-seed 0 \
  --output prompts/wikitext-2-train-articles-mamba-1024-100.jsonl
```

The checked-in file was originally materialized without requested revision
arguments (its records truthfully retain those fields as null) and resolved to
the same commits shown above. The command therefore reproduces the text and
source selection, but its requested-revision metadata is intentionally not
byte-identical.

WikiText is derived from verified Wikipedia articles and is distributed under
the Creative Commons Attribution-ShareAlike and GNU Free Documentation
licenses. Dataset card: <https://huggingface.co/datasets/Salesforce/wikitext>.

These texts are conditioning inputs only. The canonicity metric excludes every
prompt token and evaluates only model-sampled continuation token IDs.
