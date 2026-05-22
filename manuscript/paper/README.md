# Springer Manuscript Draft

This folder contains the Springer Nature LaTeX manuscript draft for the PCOS SSL project.

Source files:

- `main.tex`: single-file manuscript draft following the `sn-jnl` class structure.
- `references.bib`: manuscript bibliography.
- `sn-jnl.cls` and `sn-mathphys-num.bst`: Springer template support files copied from the December 2024 package.

Build command used locally:

```bash
tectonic main.tex
```

Current draft status:

- Full first-pass paper structure is written.
- Tables use current report values from `reports/`.
- Figures are referenced from `reports/figures/`.
- Author details, affiliations, funding, and final journal-specific declarations still need replacement before submission.
