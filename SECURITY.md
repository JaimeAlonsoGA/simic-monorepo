# Security policy

simic records meetings — other people's voices included — so a vulnerability can
expose personal data. Please report one privately, not in a public issue.

## Reporting

Use GitHub's private reporting: **Security → Report a vulnerability** on
[the repository](https://github.com/JaimeAlonsoGA/simic-monorepo/security/advisories/new).
Include what an attacker could do, and the steps or a proof of concept.

You will get an answer within a week. A confirmed issue is fixed in a release, and you
are credited in the advisory unless you prefer otherwise.

## Supported versions

Fixes go into the latest release only.

## In scope

- Recordings, transcripts or credentials readable by another local user, or written
  where they should not be (simic keeps recordings in a private `0700` directory).
- Meeting data leaving the machine on a path the user did not opt into. By default
  nothing does: remote transcription (`SIMIC_STT_BACKEND`) and LLM summaries are
  opt-in, and summaries can mask personal data (`SIMIC_MASK_PII`).
- API keys or tokens leaking into logs, run records or error messages.
- Command injection through file names, session names or settings.
