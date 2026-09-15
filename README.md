# Placement prep

Two self-contained HTML pages. No build step, no dependencies — open either file
in a browser, or view them live via GitHub Pages.

## Drill Room — `Drill_Room.html`

Daily practice for the **Zifo** and **Zoho** assessments.

**Zifo**
- Aptitude sprint — 40 questions in 30 minutes, the real elimination format
- Aptitude coach — 60s per question with the worked solution straight after
- Multitasking round — 13 levels in 20 minutes, mirroring Zifo's gamified section
- Science section — molecular biology, genetics, biochemistry, cell biology, bioinformatics
- Essay — two images, 600 words, 20-minute clock
- Full mock — all four stages end to end

**Zoho**
- Rapid fire — output prediction, complexity, data structures
- DSA — 20 repeat-offender problems with approach and complexity
- SQL — one schema, 15 queries

Tracks a daily streak, per-topic accuracy, and resurfaces questions you got wrong.
Progress is stored in the browser's localStorage, so it stays on whichever device
you use.

## Deep Learning notes — `Deep_Learning_Units_II_III_Notes.html`

BIN318 Units II and III, in syllabus order, with three interactive demos:

- **Activation explorer** — plots g(x) against g'(x) so the vanishing-gradient
  region is visible
- **Convolution playground** — editable input grid, swappable kernels, plus an
  output-size calculator for `(W − K + 2P)/S + 1`
- **Optimizer race** — SGD vs momentum vs Adam on an ill-conditioned loss surface

Covers activations, loss functions, backpropagation, regression and classification
designs, optimizers, convolution, pooling, CNN architecture, random and unsupervised
features, RNNs, BPTT, LSTM, GRU, time-series forecasting and bidirectional RNNs.

## Running them

Either open the `.html` files directly from disk, or enable GitHub Pages
(Settings → Pages → Deploy from a branch → `main` → `/ (root)`) and use the
published URLs.
