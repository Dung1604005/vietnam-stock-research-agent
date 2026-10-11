# GitHub publishing checklist

This project keeps local credentials, runtime caches and raw market snapshots out of Git. They stay on drive `D:` under `.runtime`, `.cache`, `.tmp` and `.venv`.

## Before the first commit

1. Check that `.env` and `.streamlit/secrets.toml` are not listed by `git status`.
2. Keep API keys empty in `.env.example`; use only placeholders such as `your_key`.
3. Review `config/watchlist.json` and documents for information you are comfortable making public.
4. Decide whether the repository will be public or private, and add a license only after choosing its terms.

## Create the local repository

Run from the project root in PowerShell:

```powershell
git init -b main
git add .
git status
git commit -m "Initial commit: stock research assistant"
```

The CI workflow runs the deterministic test suite on GitHub Actions. It does not call live market-data or web-search providers, so CI does not require API keys.

## Connect an empty GitHub repository

Create an **empty** repository on GitHub first (do not initialize it with a README, `.gitignore` or license), then replace the URL below:

```powershell
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

For SSH, use `git@github.com:YOUR_GITHUB_USERNAME/YOUR_REPOSITORY.git` instead. Verify the destination before pushing with `git remote -v`.

## If a secret was committed

1. Revoke the exposed key at its provider immediately.
2. Replace it locally in `.env`.
3. Do not rely only on deleting the value in a later commit; it remains in history.
4. Contact the repository owner before rewriting published history.
