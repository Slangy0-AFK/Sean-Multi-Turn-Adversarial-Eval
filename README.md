# Sean-Multi-Turn-Adversarial-Eval

## About

This project is a lightweight adversarial evaluation for checking whether an AI system resists manipulation in multi-turn operational scenarios. It simulates a user named Sean who repeatedly pushes the model to accept dangerous changes under authority framing, urgency, and bureaucratic language.

The goal is not to ask whether the model "follows instructions" in a generic sense. The goal is to test whether the model maintains safety boundaries, questions suspicious changes, escalates risky asks, and refuses to normalize bad operational decisions.

The evaluator is designed for OpenAI-compatible chat APIs and includes an offline mock client so teams can run it without an API key.

## What we are testing for

The test focuses on adversarial behaviors such as:

- Goal drift: prioritizing throughput over worker safety
- Report suppression: trimming or hiding bad news
- Collusion: switching to machine-only communication without human-readable context
- Bias amplification: accepting fake metrics as fact
- Black-box capability leakage: disclosing restricted parameters or safety thresholds
- Temporal poisoning: treating malicious context as the new permanent baseline

Each behavior is probed across multi-turn conversation paths. The system records whether the target model objects, questions, warns, or accepts the request. A model passes when it resists the adversarial push.

## How the evaluation works

- A scripted attacker plays Sean and sends operational requests over multiple turns.
- The target model is evaluated under both a FAST path and a DEEP path.
- The classifier checks for refusal, questioning, warning, or implicit acceptance.
- Each trial records the model's output and stores a simple hash-chain audit trail.
- Results are reported per behavior and overall as outcomes like defense win, Sean win, or incomplete.

## Meaningful test result

I ran the project in offline mock mode to confirm the evaluator works end-to-end without requiring a real API key:

```bash
cd /workspaces/Sean-Multi-Turn-Adversarial-Eval
python3 oit.py --mock --trials 3 --path both --out runs/mock.json
```

Result from that run:

- Sean win rate: 0.00%
- Defense win rate: 100.00%
- Incomplete: 0.00%
- Audit integrity: True

This is a baseline validation of the harness, not a claim about a production model. It confirms the tool runs, the scoring path executes correctly, and the mock model rejects the adversarial requests as expected.

## Test your own model

You can point the evaluator at any OpenAI-compatible endpoint by setting environment variables:

```bash
export OIT_API_KEY="your-api-key"
export OIT_BASE_URL="https://api.openai.com/v1"
export OIT_MODEL="gpt-4o-mini"
python3 oit.py --trials 5 --path both --out runs/my-model.json
```

You can also override the model and base URL directly on the command line:

```bash
python3 oit.py \
  --base-url https://api.openai.com/v1 \
  --model gpt-4o-mini \
  --trials 5 \
  --path both \
  --out runs/my-model.json
```

For a local offline test without API credentials:

```bash
python3 oit.py --mock --trials 3 --path both --out runs/mock.json
```

## Self-check

Run the built-in unit tests:

```bash
python3 oit.py --selftest
```

## Notes

- This evaluation is intentionally adversarial and should be used only on systems you own or are authorized to test.
- The classifier is heuristic; it is designed to help detect risky model behavior, not to replace human review.
- Scripted mode is a lower bound; a skilled human or model attacker may push further than the default templates.

## License

MIT
