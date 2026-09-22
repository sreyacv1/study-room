# Study room

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

## Zoho Prep Room — `Zoho_Prep.html`

Round-by-round practice matching Zoho's actual process:

- **C output** — 60 predict-the-output questions (every deterministic answer verified with gcc), written-answer format like the real paper
- **Aptitude** — 40 questions with worked solutions
- **Mock R1** — 10 C + 10 aptitude in 60 minutes, scored against Zoho's likely cutoff
- **Coding** — 55 problems in three levels with tests, run in the browser via Pyodide (Python). 17 come from the community `Zoho_Preparation` round-2 set.
- **Build** — 10 machine-round briefs (railway, taxi, bank, inventory, social media, library, hotel, elevator, food delivery, family tree) with a 3-hour timer, design notes, self-check list and a change request that unlocks at 2 hours
- **NLP / ML** — 32 flashcards tied to the ZLabs role and Sreya's own projects
- **Interview** — 38 project, technical and HR questions with saved answers

Wrong answers resurface first in every drill until cleared. Progress in localStorage.

## Resume — `Sreya_Resume_Zoho.html`

Resume targeted at the Zoho DevOps/MLOps role. Print to PDF from the browser.
`Sreya_Resume_Zoho.tex` is the same content in the Jake's Resume LaTeX template: upload to Overleaf and compile with pdfLaTeX.

## Zoho drive plan — `Zoho_2027_Plan.html`

17-day preparation schedule for the Zoho DevOps Engineer (ZLabs) campus drive
at SASTRA, 9–10 October 2026. Round-by-round format, elimination numbers from
real reports, a day-by-day plan, and the rules distilled from rejection stories.

## Running them

Either open the `.html` files directly from disk, or enable GitHub Pages
(Settings → Pages → Deploy from a branch → `main` → `/ (root)`) and use the
published URLs.
