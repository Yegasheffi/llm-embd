# Publishing embd on GitHub

This checklist is for maintainers. It assumes the repository lives at
`https://github.com/<owner>/<repo>` and that releases are cut from `main`.

## 1. One-time repository setup

1. Create an empty public repository (suggested name: **`embd`**).
2. Set the local remote:

   ```bash
   git remote add origin git@github.com:<owner>/<repo>.git
   git branch -M main
   git push -u origin main
   ```

3. Set the GitHub description:

   > Local incremental embedding daemon for RAG — mtime + content hash +
   > content-defined chunk alignment. SQLite, HTTP API, zero external services.

4. Add topics:

   ```text
   rag, embeddings, vector-search, incremental-indexing, fastcdc, sqlite,
   local-first, semantic-search, embedding-cache, python
   ```

5. Upload `docs/social-preview.png` under **Settings → General → Social preview**.

6. Add the owner-specific public links:

   - Add the CI badge below under the title in `README.md`.
   - Add a `[project.urls]` block to `pyproject.toml` (optional but recommended):

     ```toml
     [project.urls]
     Repository = "https://github.com/<owner>/<repo>"
     Issues = "https://github.com/<owner>/<repo>/issues"
     Changelog = "https://github.com/<owner>/<repo>/blob/main/CHANGELOG.md"
     ```

   Recommended CI badge to place under the title in `README.md`:

   ```markdown
   [![CI](https://github.com/<owner>/<repo>/actions/workflows/ci.yml/badge.svg)](https://github.com/<owner>/<repo>/actions/workflows/ci.yml)
   ```

7. Recommended repository settings:

   - **Actions → General**: allow actions from GitHub, and allow the workflow
     token to create releases (`Read and write permissions`).
   - **Branches → main**: require the `CI` status checks before merging.
   - Enable **Private vulnerability reporting** (Security tab).
   - Enable **Dependabot alerts** and **Dependabot security updates**.
   - Enable **Secret scanning** and **Push protection** (public repositories).
   - Enable Issues and, if you want open-ended Q&A, Discussions.

8. Set your Git identity before making further commits:

   ```bash
   git config user.name "Your Name"
   git config user.email "you@users.noreply.github.com"
   ```

   If the initial commit was made by a placeholder identity, rewrite it before
   the first push, for example:

   ```bash
   git commit --amend --reset-author --no-edit
   git tag -d v0.1.0
   git tag -a v0.1.0 -m "embd v0.1.0"
   ```

## 2. Every release

Versioning follows SemVer. The git tag must be `v` + `pyproject.toml` version
(the release workflow enforces this).

1. Update `version` in `pyproject.toml`.
2. Update `embd/__init__.py` `__version__`.
3. Move `CHANGELOG.md` entries from `[Unreleased]` into a dated version section.
4. Update the `**Status:** vX.Y.Z` line and the tests badge in `README.md`.
5. Run the preflight and full test suite:

   ```bash
   python3 -m unittest discover -v
   python3 scripts/preflight.py
   python3 -m build
   python3 -m embd bench -n 60
   ```

6. Commit and push:

   ```bash
   git add -A
   git commit -m "release: v0.1.0"
   git push origin main
   ```

7. Tag and push the tag. This prepared checkout already has an annotated
   `v0.1.0` tag locally, so for the first release just push it:

   ```bash
   git tag -n                 # verify the local tag
   git push origin v0.1.0
   ```

   If you committed owner-specific links after this prepared tag, move the tag
   to the final commit first:

   ```bash
   git tag -d v0.1.0
   git tag -a v0.1.0 -m "embd v0.1.0"
   git push origin v0.1.0 --force
   ```

   For later releases, create a new tag first:

   ```bash
   git tag -a v0.1.1 -m "embd v0.1.1"
   git push origin v0.1.1
   ```

8. The **Release** workflow builds `sdist`/`wheel`, writes `SHA256SUMS`, and
   creates a GitHub Release with generated notes. Check the Actions run and the
   release assets.

9. Smoke-test the published wheel in a clean environment:

   ```bash
   python3 -m venv /tmp/embd-release
   /tmp/embd-release/bin/pip install <path-or-url-to-wheel>
   /tmp/embd-release/bin/embd doctor
   /tmp/embd-release/bin/embd bench -n 20
   ```

## 3. Web-only publishing (no local git)

If you are creating the repository through the GitHub web UI:

1. Create an empty repository; do **not** initialize it with a README,
   `.gitignore`, or license, because those files are already in the upload set.
2. Open the upload folder, press **Cmd + Shift + .** on macOS to reveal hidden
   files, then select the **contents** of the `repo/` folder (not the `repo/`
   folder itself) and drag them into **Add file → Upload files**. Commit as
   `Initial public release: embd 0.1.0`.
3. Verify the repository root contains `embd/`, `tests/`, `README.md`, and the
   hidden `.github/`, `.gitignore`, `.gitattributes`, `.editorconfig`.
4. Create a release with tag `v0.1.0` in the web UI.
5. Go to **Actions → Release → Run workflow**, enter `v0.1.0`, and run it. It
   builds the wheel/sdist and attaches them to the release. The release page
   also accepts the prebuilt files in `release-assets/` if Actions are
   unavailable.
6. Continue with the repository settings in section 1.

Web uploads on GitHub do not preserve the executable bit. The documented
commands use `python3 -m embd` / `bash scripts/demo.sh`, and preflight only
warns about missing modes.

## 4. Before every push

```bash
python3 scripts/preflight.py
python3 -m unittest discover -v
git diff --check
git status --short
```

`scripts/preflight.py` checks version consistency, required release files,
oversized files, generated artifacts, accidental local paths, and common secret
patterns. It is also run in CI.

## 5. Do not publish

- `~/.embd/index.db*` or any index database (contains extracted text),
- `.env`, API keys, tokens, private keys, or provider credentials,
- personal documents used for testing,
- `build/`, `dist/`, `*.egg-info/`, `__pycache__/`.

The repository `.gitignore` covers these. Preflight fails if they exist in the
working tree (except ignored build directories such as `dist/` and `build/`) or
are tracked by git.
