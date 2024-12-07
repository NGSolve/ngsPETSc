# Poetry notes for Umberto

## Poetry
Remember that poetry cannot be installed in the same environment as the project.
Create a virtual environment somewhere permanent and you can use a bash alias to drive poetry:

```bash
python -m venv ~/.poetry
source ~/.poetry/bin/activate
pip install poetry

# check location of executable with
which poetry

# create alias (could be added to ~/.bash_profile)
alias poetry="$HOME/.poetry/bin/poetry"
```

When carrying out any of the below tasks create a new empty virtual environment!
```bash
python -m venv temp
source temp/bin/activate
```

## Add a dependency
[Docs](https://python-poetry.org/docs/cli/#add)
```bash
poetry add numpy@^2
```

## Build ngsPETSc
[Docs](https://python-poetry.org/docs/cli/#build)
```bash
poetry build
```

## Create a release
Follow these steps to create a release:
```bash
# For a bug fix:
poetry version patch
# OR for a minor version bump:
poetry version minor
# OR for a major release:
poetry version minor

git add pyproject.toml
git commit -m "Release v$(poetry version)"

git tag -a -m "Release v$(poetry version)" v$(poetry version)

git push
git push origin v$(poetry version)
```
