# Style Guide - pmmlruntime mdBook

This file is not in SUMMARY.md but every writing agent must follow it verbatim.
The quickstart page is the worked example for layout and voice.

## Voice

- Second person, imperative: "You can", "Run", "Load". No "we" except "pmmlruntime supports".
- Sentences 12-18 words, paragraph 2-3 sentences before a bullet/code.
- Bold defined terms on first use: **Session**, **Batch**, **MiningSchema**.

## Page scaffold (mandatory order)

1. H1 `# Title`: sentence case after the first word, one per page.
2. Audience callout first: `> **Info:** Looking for ...? See [Other guide].` For example, JVM users go to the migration page and training or export goes to the converter links.
3. Intro paragraph: who it is for, plus `you will learn` bullets with **bold verbs**. Keep it to 4 bullets.
4. H2 sections with `##`, which become the right-hand "On this page" TOC. Tutorials use numbered `## Step 1: ...` through Step 6.
5. Per step, keep the rhythm: prose, code block, what the result means, a visual with a caption, then a cross-link. Never place two code blocks back to back without prose between them.
6. Concept pages start with `## Concepts` and a `Concept | Description` table, with **bold terms** on first use.
7. Model pages run `Overview`, `How to export`, `How to score`, `PMML knobs`, `Fixture`.
8. Code block every 4 paragraphs or fewer.
9. At least one admonition per page: `> **Note:**`, `> **Warning:**`, `> **Tip:**`, `> **Info:**`, `> **Attention:**`.
10. Tables for outputs: after a scoring example show a small result table, not only a code dump.
11. `## Next Steps` last: one short closing sentence, exactly 4 links, and the pagination line.
12. `## FAQ` on production and migration pages only, and only when it answers real questions.

## Code blocks

- Use ```rust /```python / ```bash /```xml / ```toml
- Runnable: imports first, then `let env = PmmlEnv::new();` preamble always shown.
- Use `DecisionTreeIris.pmml` / `GradientBoosterTest.pmml` as the canonical fixtures.
- After block: 1-2 sentences explaining what happened + link to deeper docs.

## Tables

- Pipe markdown, header bold, first col left-aligned.
- For Why/Benefits: `|  | JPMML | pmmlruntime |` with **pmmlruntime** bold.
- For Concepts: `| Concept | Description |` 3-5 rows.

## Diagrams

- Mermaid: `graph TD/LR`, `flowchart`, `sequenceDiagram`, `classDiagram`. Palette:
  base `#0b7285`, infra `#36404a`, session `#e8590c`. Same as ARCHITECTURE.md.
- One diagram per concept-heavy page; hub has one `bytes → session → run`.
- Visuals show a centered image or diagram, descriptive alt text, and one caption sentence after it ("The ... page shows ..."). For engine internals use Mermaid instead of screenshots, with the same caption treatment.

## Links

- Internal: `[Session](./concepts/session.md)` relative, check built.
- API: link the API name as code and point at docs.rs, for example
  `[Session::from_bytes](https://docs.rs/pmmlruntime/latest/pmmlruntime/session/struct.Session.html#method.from_bytes)`. Never leave a bare path.
- No naked URLs; always [text](url).

## Length enforcement

- Setup: 1400 words; Quickstart: 1150; Concept: 1250 per page; Model overview: 1050; Model group: 1250; Transform: 1600; Deploy: 1450; Eval: 850; Production: 1700; Internals: 1450.
- These budgets are the size each page had after the 2026 rewrite, rounded up. A page may not grow past its budget without deleting something else on the page. CI compares with `wc -w docs/src/**/*.md` at plus or minus 15 percent.

## Admonitions (via blockquote until mdbook-admonish added)

- Note:    `> **Note:** By default ...`
- Info:    `> **Info:** Looking for ... See [Other Guide].`
- Warning: `> **Warning:** Values are not ...`
- Tip:     `> **Tip:** Use ... for ...`
- Attention: `> **Attention:** This feature is only ...`

## No other products (mandatory)

- A published page never names another vendor or product, and never uses "like X", "similar to X", or "X calls this ...". State the behaviour of pmmlruntime directly.
- The only allowed outside names are the converters that produce PMML (`sklearn2pmml`, `r2pmml`, `lightgbm2pmml`, `jpmml-sparkml`, `jpmml-xgboost`, `jpmml-lightgbm`), the libraries used in real examples (`scikit-learn`, `numpy`, `pyarrow`), the formats (`PMML`, `Arrow`, `CSV`), and JPMML when the page compares measured numbers.
- If a sentence only exists to explain how another tool words something, delete the sentence.

## Style check (mandatory for every agent)

- Before writing or revising a page, read `docs/src/getting-started/quickstart.md` as the worked example: callout, learn bullets, `Step N:` sections, output tables, captions, Next Steps. Match that shape and that voice.
- Read the page you are about to change and the two pages it links to, so the new text agrees with its neighbours.
- Report which page you used as the reference, and which sections you renamed or removed.

## Humanizer rule (mandatory for every agent)

- Every page you write or revise must pass the `humanizer` skill before you report done
  (`/home/pab1s/.agents/skills/humanizer/SKILL.md`). Run its full rewrite process on your prose:
  mark the AI patterns, draft, read it aloud, then answer its two checks ("what still sounds
  AI-generated?" and "did the rewrite add or drop any claim?").
- The bans that matter most here, taken from the skill:
  - §14 no em dashes or en dashes, and no spaced dash substitutes (` - `, ` -- `). Use a period,
    comma, colon, or parentheses, or rewrite the sentence. A hyphen inside a word stays. Step headings use `Step 1: Title`.
  - §7 overused words to cut unless they are API names: additionally, comprehensive, crucial,
    delve, enhance, ensure, foster, highlight, intuitive, leverage, robust, seamless, showcase,
    underscore, valuable, vibrant.
  - §3 no "-ing" analysis tails (`..., highlighting X`); §4 no sales words; §15 bold only defined
    terms; §16 no bold mini-heading lists; §17 sentence-case headings; §25 no generic positive
    ending; §29 never repeat the heading as the first sentence; §13 write active voice.
- Keep every fact: numbers, µs and ns figures, fixture names, API names, file paths, command
  flags. You may shorten or reorder, never add or drop.
- Report the humanizer pass to your suborchestrator (patterns found, sentences rewritten).

## Cross-links required

- Every page must have `## Next Steps` with 3-4 valid cross-links to other chapters.
- Every model page must link to `../../concepts/schema.md` and `../../evaluation/correctness.md`.
- Every deployment page must link to `../../getting-started/quickstart.md`.

## Book structure

- `Introduction` is the landing page and the shortest path to the library.
- `Getting Started` holds setup, the quickstart, and the converter guide (one file, any framework).
- `Core Concepts` (Session, Values, Schema) and `Batch & Data` (Batch, Arrow) explain the runtime model before any model detail.
- `Supported Models` covers the 19 model elements in four groups, and `Transforms & Engine` covers derived fields, builtins, and predicates.
- `Evaluation & Benchmarks` holds correctness, performance, and validation gates.
- `Deployment` lists the bindings overview first, then Rust, Python, C, Java, CLI, JavaScript, and Docker.
- `Production` holds security, concurrency, troubleshooting, and the JPMML migration path.
- `Internals` holds architecture, IR, engine dispatch, and the execution provider for contributors.
- `API Reference` closes the book with the public surface and docs.rs links.

## Site chrome

- Search uses the mdBook built-in search (`[output.html.search]`, `limit-results = 20`).
- Breadcrumbs come from SUMMARY nesting; keep nesting 2 levels deep or less.
- The right-hand "On this page" TOC comes from `##` H2s, so keep H2s short and stable. They are anchors.
- Prev/Next arrows follow SUMMARY order, and `Next Steps` repeats the next page explicitly.
- Dark and light themes come from mdBook; no work needed.

## Deliberate gaps

- No versioned URLs, no hosted search, no separate API subdomain, no embedded video. Add them only when mdBook itself becomes the limit.
