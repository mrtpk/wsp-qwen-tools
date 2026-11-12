import re, sys, markdown

src, out = sys.argv[1], sys.argv[2]
md = open(src).read()
md = md.split("\n", 1)[1]  # drop the H1; the page has its own header
md = re.sub(r"^Status: .*\n", "", md, count=1, flags=re.M)
md = re.sub(r"^  - ", "    - ", md, flags=re.M)
body = markdown.markdown(md, extensions=["tables", "sane_lists"])

toc = []
def h2(m):
    text = re.sub(r"<[^>]+>", "", m.group(1))
    slug = "s" + text.split(".")[0].strip() if text[0].isdigit() else re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    toc.append((slug, text))
    return f'<h2 id="{slug}">{m.group(1)}</h2>'
body = re.sub(r"<h2>(.*?)</h2>", h2, body)
body = re.sub(r"<table>", '<div class="tw"><table>', body)
body = body.replace("</table>", "</table></div>")
body = re.sub(r"<td>(yes|\*\*yes\*\*|<strong>yes</strong>)</td>", '<td><span class="pill ok">kept</span></td>', body)
body = re.sub(r"<td>no</td>", '<td><span class="pill no">not kept</span></td>', body)
# Wrap the verdict section in a callout.
body = re.sub(r'(<h2 id="s1">.*?)(?=<h2 id="s2">)', r'<section class="verdict">\1</section>', body, flags=re.S)

toc_html = "\n".join(f'<li><a href="#{s}">{t}</a></li>' for s, t in toc)

page = f"""<title>Tool-Call Weight Edits</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;700;800&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
/* Layout: lab report. Sticky contents rail on wide screens, one reading column, wide tables scroll in place. */
:root {{
  --bg: #f6f7f9; --panel: #ffffff; --ink: #1b1f27; --muted: #5b6372; --rule: #d9dde4;
  --accent: #2a4bb0; --accent-soft: #e6ebf8; --ok: #1d7a4d; --ok-soft: #e1f2e8; --no: #a0412d; --no-soft: #f7e5e0;
  --code-bg: #eceef3;
  --f-display: "Archivo", "Helvetica Neue", Arial, sans-serif;
  --f-body: "Source Sans 3", "Segoe UI", system-ui, sans-serif;
  --f-mono: "JetBrains Mono", ui-monospace, Menlo, Consolas, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #12151b; --panel: #1a1e26; --ink: #e4e7ee; --muted: #9aa3b4; --rule: #2d333f;
  --accent: #8fa8ff; --accent-soft: #232c45; --ok: #6fd3a0; --ok-soft: #173326; --no: #f0957f; --no-soft: #3a211b;
  --code-bg: #232833; color-scheme: dark; }} }}
:root[data-theme="dark"] {{
  --bg: #12151b; --panel: #1a1e26; --ink: #e4e7ee; --muted: #9aa3b4; --rule: #2d333f;
  --accent: #8fa8ff; --accent-soft: #232c45; --ok: #6fd3a0; --ok-soft: #173326; --no: #f0957f; --no-soft: #3a211b;
  --code-bg: #232833; color-scheme: dark; }}
body {{ background: var(--bg); color: var(--ink); font-family: var(--f-body); font-size: 17px; line-height: 1.6;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); padding-block: 0 64px; }}
a {{ color: var(--accent); }}
a:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
header {{ max-width: 1180px; margin: 0 auto; padding-block: 48px 28px; border-bottom: 1px solid var(--rule); }}
.eyebrow {{ font-family: var(--f-mono); font-size: 12.5px; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); }}
h1 {{ font-family: var(--f-display); font-weight: 800; font-size: clamp(30px, 5vw, 46px); line-height: 1.1; margin: 10px 0 12px; text-wrap: balance; letter-spacing: -.01em; }}
.lede {{ max-width: 68ch; color: var(--muted); font-size: 18px; margin: 0; }}
.figs {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin-top: 28px; }}
.fig {{ background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; padding: 14px 16px; display: flex; flex-direction: column; gap: 4px; }}
.fig b {{ font-family: var(--f-mono); font-size: 22px; font-weight: 600; font-variant-numeric: tabular-nums; }}
.fig span {{ font-size: 14px; color: var(--muted); line-height: 1.35; }}
.wrap {{ max-width: 1180px; margin: 0 auto; display: grid; grid-template-columns: 230px minmax(0, 1fr); gap: 48px; padding-top: 32px; }}
nav {{ position: sticky; top: calc(env(safe-area-inset-top, 0px) + 20px); align-self: start; font-size: 14.5px; }}
nav ol {{ list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 6px; }}
nav a {{ color: var(--muted); text-decoration: none; }}
nav a:hover {{ color: var(--accent); }}
main {{ min-width: 0; max-width: 820px; }}
h2 {{ font-family: var(--f-display); font-weight: 700; font-size: 26px; margin: 52px 0 14px; padding-top: 14px; border-top: 2px solid var(--ink); text-wrap: balance; scroll-margin-top: 16px; }}
h3 {{ font-family: var(--f-display); font-weight: 500; font-size: 19px; margin: 32px 0 8px; color: var(--ink); }}
main > p, main li, section p, section li {{ max-width: 70ch; }}
ul, ol {{ padding-left: 1.3em; }}
li {{ margin: 4px 0; }}
code {{ font-family: var(--f-mono); font-size: .84em; background: var(--code-bg); padding: 1px 5px; border-radius: 3px; overflow-wrap: anywhere; }}
.tw {{ overflow-x: auto; margin: 16px 0 20px; border: 1px solid var(--rule); border-radius: 6px; background: var(--panel); }}
table {{ border-collapse: collapse; width: 100%; font-size: 14.5px; line-height: 1.4; }}
th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--rule); vertical-align: top; font-variant-numeric: tabular-nums; }}
th {{ font-family: var(--f-display); font-weight: 500; font-size: 13px; letter-spacing: .03em; text-transform: uppercase; color: var(--muted); background: var(--bg); white-space: nowrap; }}
tr:last-child td {{ border-bottom: 0; }}
td strong {{ color: var(--accent); }}
.pill {{ display: inline-block; font-family: var(--f-mono); font-size: 12px; padding: 1px 8px; border-radius: 999px; white-space: nowrap; }}
.pill.ok {{ background: var(--ok-soft); color: var(--ok); }}
.pill.no {{ background: var(--no-soft); color: var(--no); }}
.verdict {{ background: var(--accent-soft); border-radius: 8px; padding: 4px 22px 14px; }}
.verdict h2 {{ border-top: 0; margin-top: 12px; }}
.verdict .tw {{ background: var(--panel); }}
footer {{ max-width: 1180px; margin: 56px auto 0; padding-top: 16px; border-top: 1px solid var(--rule); color: var(--muted); font-size: 14px; }}
@media (max-width: 860px) {{
  .wrap {{ grid-template-columns: minmax(0, 1fr); gap: 8px; }}
  nav {{ position: static; border: 1px solid var(--rule); border-radius: 6px; padding: 12px 16px; background: var(--panel); }}
  nav ol {{ display: grid; grid-template-columns: 1fr 1fr; gap: 4px 16px; }}
  h2 {{ font-size: 22px; }}
  .verdict {{ padding: 2px 14px 10px; }}
}}
@media (max-width: 420px) {{ nav ol {{ grid-template-columns: 1fr; }} body {{ font-size: 16px; }} }}
</style>

<header>
  <div class="eyebrow">Qwen3.5-0.8B · BFCL-V4 · layer 23 · token <code>&lt;tool_call&gt;</code> = 248058</div>
  <h1>Weight edits for tool use: v1 and v2</h1>
  <p class="lede">Final report, 2026-11-08 (UTC+4). Two studies edit the weights of Qwen3.5-0.8B to improve its tool-call decision. v1 fits a rank-1 edit on BFCL data. v2 programs MLP neurons from practice data under the openwsp protocol.</p>
  <div class="figs">
    <div class="fig"><b>20.84 → 22.54</b><span>BFCL thinking, base → v2 v7 (3 seeds)</span></div>
    <div class="fig"><b>17.68 → 20.29</b><span>BFCL non-thinking, base → v1 E4b (2 seeds)</span></div>
    <div class="fig"><b>6,144</b><span>weights changed by v7 (E4b: about 3.5M)</span></div>
    <div class="fig"><b>far below 60</b><span>the stretch target; decision edits add 1 to 2.6 points</span></div>
  </div>
</header>

<div class="wrap">
  <nav aria-label="Contents"><ol>
{toc_html}
  </ol></nav>
  <main>
{body}
  </main>
</div>
<footer>Sources: <code>v1/docs/</code> (F1–F49, D1–D18) and <code>v2/docs/</code> (G1–G35, V2-D1–V2-D13). Web search is not run; overall scores count it as 0.</footer>
"""
open(out, "w").write(page)
print(len(page), "bytes;", len(toc), "sections")
