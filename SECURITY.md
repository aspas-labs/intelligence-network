# Security policy

## Reporting a vulnerability

Please do **not** open a public issue for a vulnerability.

Use GitHub's private reporting: **Security → Report a vulnerability** on this
repository. Include the affected version or commit, a description, and, if you
can, steps or a test that reproduce it.

Maintainers: enable *Private vulnerability reporting* in the repository
settings so the button above exists.

## What to expect

- Acknowledgement of your report, then an assessment and a fix plan.
- Credit in the advisory if you want it.
- Please give maintainers a reasonable time to ship a fix before publishing details.

## Supported versions

Only the latest release and `main` receive security fixes while the project is
pre-adoption.

## Scope

In scope: the wire protocol and its parsers, the QUIC transport and identity
binding, discovery and the DHT, the job runtime and bubblewrap sandbox,
artifact storage, distributed training, the installers and release pipeline.

Out of scope: results that depend on already having local access to the
node's account, and findings that only affect the labelled `EMULATED` output.

The current security posture and known limits are described in
[docs/SECURITY.md](docs/SECURITY.md).
