
# Notes for Developers

## 1. Setting up the project

### A. Running inside container (preferred)

Build the docker image:

```bash
docker build --tag 'ai-dio:dev' .
```

Run the image with necessary flags for graphics forwarding:

```bash
docker run -it \
  --env DISPLAY=$DISPLAY \
  --env XAUTHORITY=$XAUTHORITY \
  -v $XAUTHORITY:$XAUTHORITY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  'ai-dio:dev'
```

### B. Running locally
Install dev depenendencies:

```bash
uv sync --group dev
```

Run project:

```bash
uv run python3 main.py
```


## 2. Pre-commit hooks

```bash
uv run pre-commit install
```

After that, linters will run on each commit. You can also run them manually:

```bash
uv run pre-commit run --all-files
```

## 3. Pull requests

#### Please refrain from pushing onto `main`!

Every change should be submitted as a **Pull Request**. This approach enables two things:

1. Other developers can review the code
2. Automatic workflows will be ran to test and check the code

**NOTE:** Consider using understandable commit titles, for example:

```
add: saving output to file
fix: out-of-bounds bug
```

## 4. Miscellaneous

1. All application source files should be inside `/src/AI-dio/` in adequate subdirectories (f.e. `gui`, `audio`, `model`)
2. Avoid pushing images/audio files/other auxiliary files into the repository
3. Keep correct naming conventions (PEP8)
