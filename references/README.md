# Paper reference data

`paper-figure5-canon-perc-length.txt` is an unchanged copy of
`tikz/data/canon_perc_len_data.txt` from the official arXiv v2 TeX source for
[Geh et al., *Where is the signal in tokenization space?*](https://arxiv.org/abs/2408.08541v2),
downloaded on 17 July 2026 from
`https://export.arxiv.org/e-print/2408.08541v2`.

- arXiv source archive SHA-256:
  `7620a2b059ca1ae5059819900c90cbaeafece6344e647aa2407a7d73b2f6121e`
- copied data file SHA-256:
  `63b37b16a1c1b8cbaf91386a8a168439d9332b255c7f225d137b83214e49c0db`

The file contains 256-based percentages through generated length 128: its
early values change in increments of `100/256`, while later percentages cease
to be multiples of that increment, consistent with smaller integer eligible
sample denominators after early stopping. The paper prose and Figure 5 caption
establish unconditional generation and the
percentage-of-canonical-sequences statistic, but do not state the initial
sample count or operational tokenizer/EOS details. Treat those as facts from
the archived plotting data and choices of this replication, respectively.
