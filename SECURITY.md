# Security policy

## Reporting a vulnerability

Do not post API keys, credentials, personal portfolio data, or an exploitable security issue in a public GitHub issue. For a private repository, report it through the repository owner's private contact channel. For a public repository, use GitHub private vulnerability reporting if it is enabled; otherwise contact the maintainer privately.

## Secrets and market data

- Never commit `.env`, `.streamlit/secrets.toml`, raw `.runtime` snapshots, caches, or local databases.
- Commit `.env.example` only with empty or clearly fictional placeholder values.
- Revoke and replace a provider key immediately if it is committed, even if the repository is made private afterwards.
- This project is research-only: do not add broker credentials or trade-execution integrations.
