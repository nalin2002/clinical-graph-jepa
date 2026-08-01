# Clinical Graph-JEPA paper

- [Read the repository copy](clinical_jepa.pdf)
- [OpenReview record](https://openreview.net/forum?id=HXsMPubPqE)
- [Paper-to-code implementation map](../docs/PAPER_CODE_MAP.md)

> [!IMPORTANT]
> The code behind this paper is **`src/fawkes/`**, not `src/clinical_jepa/`.
> The released checkpoint is `models/fawkes-entity-note/`.

`clinical_jepa.pdf` is an exact copy of the source PDF supplied with this
workspace. Its SHA-256 checksum is:

```text
eaeb3a637b7c30a755a704f26dfd9ebf265a0028511b3cb3a2b47af02d0b997f
```

The PDF describes the clinical knowledge-graph construction, Graph-JEPA
representation learning, localized Clinical-ModernBERT note conditioning,
frozen-encoder relation recovery, and evaluation protocol. See the
implementation map before comparing model results: the suite deliberately
preserves two related implementation lineages, and their version numbers are
not one continuous sequence.
