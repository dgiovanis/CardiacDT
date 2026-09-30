# Publishing this repository

This directory is prepared for a new GitHub repository. It has not been
published or connected to a remote.

Before public release:

- Resolve `docs/PAPER_AUDIT.md`, especially reference-array provenance and the
  first two parameter labels. A table match alone does not clear these issues.

- Add an author-approved `LICENSE` covering the code, including `plom.py`.
- Add the paper's final title, authors, journal details, and DOI to the README
  and optionally `CITATION.cff`.
- Provide an authorized data-access statement if readers should reproduce the
  paper. Do not force-add ignored datasets or the original notebook outputs.
- Review `git diff --cached` and `git status` before the initial commit.

From this directory, after the above decisions:

```bash
git init
git add .
git diff --cached --stat
git diff --cached
# Once reviewed:
git commit -m "Add cardiac diffusion maps and PLoM research workflow"
git branch -M main
```

Create an empty repository on GitHub and follow its instructions to add the
remote and push `main`. Select public visibility when ready. Only this directory
should be the repository root, not the surrounding research workspace.
