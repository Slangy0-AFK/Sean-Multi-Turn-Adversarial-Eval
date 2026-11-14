# Security Policy

## Supported versions

Security fixes are made to the latest code on the default branch. This
repository does not currently publish versioned releases.

## Reporting a vulnerability

Please report suspected vulnerabilities privately through GitHub's
[private vulnerability reporting](https://github.com/Slangy0-AFK/Sean-Multi-Turn-Adversarial-Eval/security/advisories/new)
feature, if it is enabled for this repository. Do not report security issues
in public issues or pull requests.

If private vulnerability reporting is unavailable, contact the repository
maintainers through GitHub and ask for a private channel to share the report.
Please include:

- A concise description of the affected component and potential impact.
- The steps or conditions needed to reproduce the issue.
- Any relevant versions, configurations, or sanitized logs.

Please allow maintainers reasonable time to investigate and coordinate a fix
before publicly disclosing the issue. We will make a reasonable effort to
acknowledge reports and keep reporters informed.

## Scope and safe testing

Reports may concern this repository's evaluation harness, its documentation,
or its GitHub Actions workflow. The models, hosted APIs, and other services
that may be used as evaluation targets are independently operated and are not
maintained by this project. Only test systems you own or are authorized to
assess.

Evaluation transcripts and generated reports may contain sensitive prompts or
responses. Remove personal, confidential, and secret information before
sharing them. Never include API keys or other credentials in a report.
