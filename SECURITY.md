# Security policy

## Supported versions

embd is pre-1.0. Security fixes are applied to the latest `0.1.x` release and to
`main`.

| Version | Supported |
|---|---|
| 0.1.x | ✅ |
| < 0.1 | ❌ |

## Reporting a vulnerability

Please do **not** open a public issue for a vulnerability.

Use GitHub's private vulnerability reporting on this repository:
**Security → Report a vulnerability**. If that is unavailable, contact the
maintainers through their GitHub profile and ask for a private channel.

Please include:

- affected version / commit,
- operating system and Python/SQLite versions,
- a minimal reproduction,
- impact and any known mitigation,
- whether the daemon was bound to localhost or a network interface.

We will acknowledge a report on a best-effort basis, investigate, and coordinate
disclosure. Please give us a reasonable window before publishing details.

## Security model and scope

embd is designed as a **local, single-user daemon**:

- It binds to `127.0.0.1` by default.
- `/v1/*` can require an API key, but there is no multi-tenant isolation.
- The API can index any path the daemon process can read. That is intentional
  for local use; do not expose it to untrusted networks.
- Documents and query text are sent to a remote embedding provider only when
  one is explicitly configured (`openai`); `hash` and local Ollama do not send
  them off-machine.
- The web console is static and served without authentication; protect the
  daemon port if the machine is shared.

In scope:

- authentication bypass of a configured API key,
- path handling that escapes the documented "index what the process can read"
  behavior in a surprising way,
- SQL injection, command injection, or template injection,
- denial of service caused by malformed API input,
- disclosure of API keys or document contents in logs.

Out of scope:

- running the daemon on `0.0.0.0` without an API key and without a firewall,
- documents containing malicious instructions that an LLM layer later follows,
- vulnerabilities in third-party embedding providers or models.

## Hardening checklist

- Keep the default `127.0.0.1` bind.
- Set `api_key` before binding to a LAN interface.
- Run the daemon as an unprivileged user with access only to intended folders.
- Keep `.embd/index.db` private; it contains extracted document text and
  embeddings.
- Use `EMBD_HOME` to isolate indexes for different projects.
