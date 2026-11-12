---
name: unattended-run
description: Run a long experiment series while the user is away, with logs, findings, decisions, and short status replies. Use when the user says "run unattended", "I will be back in N hours", "continue the experiments", or asks "how is it?" or "progress?" during a long run.
---

# Unattended experiment runs

## Before the user leaves

Get these from the user. Do not invent them.

- The goal and the stop rule (a score, a time, a count of runs).
- The permissions (GPU, downloads, installs, deletes).
- Any limit. Do not add a limit of your own. If a protocol has a limit, show it and ask.
  (A run once had a pass budget that the user never set: "I never set a budget".)

## During the run

- Run every command in one tmux session. Write each job's output to a log file.
- Write a `.done` marker file when a job ends. Do not poll with short sleeps.
- Check that a long job's log grows. A hung job with an idle server is common. Restart it.
- Get each time from `date`. Never estimate a time in a log.
- Keep three files and append to them:
  - `progress-log.md`: time, action, result. One line per event.
  - `findings.md`: numbered findings (G1, G2, ...), each with the numbers and the source file.
  - `decisions.md`: numbered decisions with the reason and the rejected options.
- Record a negative result as a finding. Record each result that is not kept, and why.
- Do not change folders that the user marked read only.
- Before you choose between two close results, check the noise (seeds). Run one more seed if needed.

## Status reply ("how is it?", "progress?")

Use this form. Keep it under ten lines.

```
State: <running X | waiting for Y | done>
Last result: <number, compared with the base and the best so far>
Next: <the next step and its ETA from date>
Problem: <none | the problem and what you did>
```

## At the end

- Write the final state in the progress log and the findings.
- Give the verdict first: the best result, the kept edit or model, and the gap to the goal.
- List what did not work, the open items, and the disk use of the work folders.
- Do not delete anything without a request.
