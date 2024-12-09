# Reef Life Survey volunteer data validator

Streamlit app for validating data entry by [Reef Life Survey](https://yanirseroussi.com/tags/reef-life-survey/) volunteers.

## Development Setup

See `.devcontainer/devcontainer.json`: This can run in a GitHub Codespace or locally via VSCode or JetBrains Gateway.

## Deployments

- The main repo includes deployment to a free (and slow) Render instance as part of the GitHub Actions flow.
  URL: https://rls-validator.onrender.com/.
- [The yanirs-streamlit fork](https://github.com/yanirs-streamlit/rls-validator/) deploys to the public Streamlit Cloud,
  which includes a more generous (and fast) free instance.
  URL: https://rls-validator.streamlit.app/.

  This is a fork by a dedicated user because Streamlit asks for excessive permissions to unrelated repos.
  Therefore, to deploy to Streamlit Cloud, the fork must be updated.

## Upgrading Python

This is a bit involved and rare, so worth documenting:
1. Update the version in `.devcontainer/devcontainer.json`.
2. Update the version in `pyproject.toml`.
3. Test locally.
4. Update the version in `lint-and-test.yml`.
5. Push to main repo.
6. On Render, update the `PYTHON_VERSION` env var (requires [minor version](https://docs.render.com/python-version), 
   which can be found [on the Python website](https://www.python.org/downloads/))
7. Test the Render deployment.
8. Update the yanirs-streamlit fork.
9. On [Streamlit](https://share.streamlit.io/), delete and re-deploy the app with the new Python version.
10. Test the Streamlit deployment.
