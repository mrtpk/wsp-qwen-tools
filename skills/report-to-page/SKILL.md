---
name: report-to-page
description: Turn a long Markdown study report into one HTML page with a contents rail, a verdict box, key figures, and kept/not-kept labels in tables, then publish it as an Artifact. Use when the user asks to publish, share, or show a Markdown report as a page.
---

# Markdown report to a page

Script: `build_report_page.py SRC.md OUT.html` (needs the Python package `markdown`).

## Steps

1. Load the `artifact-design` skill if the Artifact tool is present.
2. Make sure the report has numbered H2 sections (`## 1. Verdict`, `## 2. ...`).
   The script makes the contents rail from them. Section 1 becomes the verdict box.
3. Edit the `page` header block in the script for this report:
   the eyebrow line, the title, the lede, and four figure boxes (number + one line).
   Take each number from the report. Do not invent a number.
4. Table cells `yes` and `no` become "kept" and "not kept" labels. Use them only in a
   "kept" column. Change the regex if the report uses other words.
5. Run: `python3 build_report_page.py REPORT.md page.html`. It prints the size and the
   section count. Check that the count equals the H2 count.
6. Publish `page.html` with the Artifact tool (icon `report`). Republish with the same path
   after each report change, so the URL stays the same.

## Page properties

- Light and dark colors as tokens on `:root`. Dark mode by media query and by `data-theme`.
- Wide tables scroll inside their box. No horizontal page scroll at phone width.
- The contents rail is sticky on wide screens and a two-column list below 860 px.
- Fonts from Google Fonts only. No scripts.
