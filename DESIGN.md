# Design System: Epistemic Prism (`apart-epistemics`)

**Committed Aesthetic Direction:** **Editorial Quantitative Lab Notebook** (Part C fallback per `claude-design-bridge` while keeping `design-briefs/epistemic-prism.md` ready for optional Claude Design import).

## Design Tokens
- **Surface & Paper:**
  - `--bg-canvas`: `#FAF8F4` (warm archival paper)
  - `--bg-card`: `#FFFFFF` (specimen card)
  - `--bg-subtle`: `#F2EEE6` (inset callout / table header)
  - `--border-hairline`: `#DED8CC`
  - `--border-strong`: `#B8B0A0`
- **Typography & Ink:**
  - `--ink-primary`: `#171614` (carbon ink, contrast > 14:1 on `#FAF8F4`)
  - `--ink-secondary`: `#545047` (slate graphite, contrast > 6.5:1)
  - `--ink-muted`: `#6E695E`
  - `--font-display`: `'Newsreader', 'Georgia', 'Times New Roman', serif`
  - `--font-sans`: `'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif`
  - `--font-mono`: `'JetBrains Mono', 'SFMono-Regular', Menlo, Consolas, monospace`
- **Semantic & Experimental Palette:**
  - `--accent-vermilion`: `#C84B31` (primary focus / monolithic collapse / alert)
  - `--accent-cerulean`: `#245A7A` (cultural/regional heterogeneity / info)
  - `--accent-moss`: `#2F6F4E` (well-calibrated / independent / healthy baseline)
  - `--accent-ochre`: `#A96A0B` (personalized peer / sycophancy caution)
  - `--accent-plum`: `#6B4C7A` (multi-lab frontier mix)
- **Layout & Motion:**
  - Max content width `1240px`, generous editorial margins, asymmetrical 7:5 hero summary/quadrant grid on desktop (`>= 960px`), single-column responsive stack at `<= 768px` (verified at `1440px` and `390px`).
  - Subtle `180ms ease-out` hover/tab transitions respecting `@media (prefers-reduced-motion: reduce)`.

