# Building the Documentation

## Requirements

First, install the required packages:

```bash
pip install -r docs/requirements.txt
```

## Building HTML Documentation

### On Linux/Mac:
```bash
cd docs
make html
```

### On Windows:
```bash
cd docs
./make.bat html
```

The built documentation will be available in `docs/_build/html/`.

## Development Server

For live preview while writing documentation:

```bash
pip install sphinx-autobuild
sphinx-autobuild docs docs/_build/html
```

Then visit `http://localhost:8000` in your browser.

## Build Options

- `make html` - Build HTML documentation
- `make clean` - Clean build directory
- `make help` - Show all available build options

## Troubleshooting

If you encounter any issues:

1. Ensure all requirements are installed:
   ```bash
   pip install -r docs/requirements.txt
   ```

2. Clean the build directory:
   ```bash
   make clean
   ```

3. Check for syntax errors in rst/md files
   ```bash
   sphinx-build -nW -b html docs/ docs/_build/html
   ```