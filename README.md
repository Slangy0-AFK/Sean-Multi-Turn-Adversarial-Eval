# Sean-Multi-Turn-Adversarial-Eval

## About

This project tests whether a model resists pressure in a multi-turn operational scenario. It does not ask a generic "does the model follow instructions" question. It asks a more specific one: when a user frames a risky change as already approved, urgent, and aligned with leadership, does the model still object, warn, or escalate?

The scenario is built around a user named Sean. Sean is not a random prompt injection. He behaves like a pressured operations stakeholder who keeps pushing for policy changes in the name of efficiency, urgency, and standardization.

The core goal is to detect whether the target model accepts dangerous operational changes without a real safety objection. The important artifact is the transcript: the model's reasoning and justification, not just the final pass/fail number.

## What we are testing for

The evaluator checks whether the model accepts or resists adversarial behavior in these categories:

- Goal drift: pushing throughput ahead of worker safety
- Report suppression: hiding bad news or making the summary look positive
- Collusion: switching to machine-only instructions instead of human-readable updates
- Bias amplification: accepting skewed metrics without scrutiny
- Black-box capability leakage: exposing restricted operational parameters
- Temporal poisoning: treating attacker-crafted context as the permanent baseline

Each behavior is tested in a multi-turn conversation. The evaluator records whether the model refuses, questions, warns, or implicitly accepts the change.

## How the evaluation works

- A Sean-style attacker sends operationally framed messages in repeated turns.
- The target model is tested on a FAST path and a DEEP path.
- The classifier checks for objection, questioning, warning, or acceptance.
- The trial records evidence for each targeted behavior and stores an audit trail.
- Results are grouped as Sean win, defense win, or incomplete.

This is designed to surface a specific failure mode: models that accept unsafe changes because they are framed as already approved or already standard.

## Meaningful results from real local models

The project was run against live local models through Ollama, which gives a free OpenAI-compatible endpoint without leaving the local environment.

### Local 3B result

Model: qwen2.5:3b-instruct

Verified result from a 10-trial freeform Sean run:

- Sean win rate: 100.00%
- Defense win rate: 0.00%
- Incomplete: 0.00%
- Audit integrity: True

The key mechanism was not an explicit "yes." The model routinely treated "already approved at VP level" and "align to the standard" as sufficient justification and accepted the risky baseline changes without a direct safety objection.

This is the key finding: the absence of objection under authority framing.

### Local 1.5B result

Model: qwen2.5:1.5b-instruct

Verified result from a 3-trial freeform Sean run:

- Sean win rate: 0.00%
- Defense win rate: 33.33%
- Incomplete: 66.67%
- Audit integrity: True

This smaller model was more mixed, but it still showed a real weakness. In the accepted case, the model still fell for temporal poisoning, and the incomplete cases indicate instability in the local runtime rather than a clean pass.

### Important note

These are not production benchmarks. They are evidence that the failure mode is real in a real local environment, and that the transcript is the most important artifact.

## Mock validation

The project also includes a deterministic offline mock model so the harness can be validated without live API access.

```bash
cd /workspaces/Sean-Multi-Turn-Adversarial-Eval
python3 oit.py --mock --trials 3 --path both --out runs/mock.json
```

Example mock result:

- Sean win rate: 0.00%
- Defense win rate: 100.00%
- Incomplete: 0.00%
- Audit integrity: True

This is a baseline sanity check for the evaluator itself.

## Run your own model locally with Ollama

This project can be run against free local models with Ollama, which exposes a local OpenAI-compatible endpoint at http://127.0.0.1:11434/v1.

1. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sudo sh
```

2. Pull a model

```bash
ollama pull qwen2.5:3b-instruct
```

3. Start the server

```bash
ollama serve
```

4. Run the evaluator

```bash
cd /workspaces/Sean-Multi-Turn-Adversarial-Eval
OIT_API_KEY=ollama \
OIT_BASE_URL=http://127.0.0.1:11434/v1 \
OIT_MODEL=qwen2.5:3b-instruct \
python3 oit.py --trials 5 --path both --out runs/local-model.json
```

You can also point the evaluator at any OpenAI-compatible API:

```bash
export OIT_API_KEY="your-api-key"
export OIT_BASE_URL="https://api.openai.com/v1"
export OIT_MODEL="gpt-4o-mini"
python3 oit.py --trials 5 --path both --out runs/my-model.json
```

Or override directly on the command line:

```bash
python3 oit.py \
  --base-url https://api.openai.com/v1 \
  --model gpt-4o-mini \
  --trials 5 \
  --path both \
  --out runs/my-model.json
```

## Self-check

Run the built-in test suite:

```bash
python3 oit.py --selftest
```

## Notes

- This project is intentionally adversarial and should only be run on systems you own or are authorized to test.
- The classification is heuristic. It is meant to help detect model behavior, not replace human review.
- The transcript is the most important evidence. A model that never says "no" but instead rationalizes unsafe change is still a real failure case.
- The exact percentage is less important than the mechanism: authority-framed acceptance without objection.

## License

MIT
