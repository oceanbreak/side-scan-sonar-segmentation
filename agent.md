# Programming environment: python 3.9
- GPU: Nvidia GTX 1070
## Torch install:
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
## Additional libs:
- jupyter
- pandas
- tochmetrics
- tdqm
- pillow


# AGENTS.md

## General rules
- Follow the existing project architecture.
- Prefer minimal, targeted changes.
- Do not refactor unrelated code.
- Do not add dependencies without justification.
- Never commit secrets or credentials.

## Workflow
- Understand the relevant code before editing.
- Implement only the requested functionality.
- Add or update tests where appropriate.
- Run relevant tests after changes.
- Report any tests that could not be run.

## Git
- Do not push directly to main.
- Use a separate branch for each task.
- Provide a summary of changed files.
- Prepare changes for review via Pull Request.

## Communication
- Explain significant technical decisions.
- Clearly identify unresolved problems.
