# Contributing Guide

## Development Setup

1. Fork and clone the repository:
```bash
git clone https://github.com/yourusername/DiffusionLM.git
cd DiffusionLM
```

2. Create a virtual environment:
```bash
python -m venv env
source env/bin/activate  # Linux/Mac
.\env\Scripts\activate   # Windows
```

3. Install development dependencies:
```bash
pip install -e ".[dev]"
```

## Code Style

We use:
- Black for code formatting
- isort for import sorting
- flake8 for linting

Run the formatters:
```bash
black .
isort .
flake8 .
```

## Running Tests

```bash
pytest tests/
pytest tests/ -v  # verbose mode
pytest tests/test_model.py  # specific test file
```

## Pull Request Process

1. Create a new branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and commit:
```bash
git add .
git commit -m "feat: add new feature"
```

3. Push changes:
```bash
git push origin feature/your-feature-name
```

4. Open a Pull Request on GitHub

## Commit Message Format

Follow conventional commits:
- feat: New feature
- fix: Bug fix
- docs: Documentation
- style: Formatting
- refactor: Code restructuring
- test: Testing
- chore: Maintenance

Example:
```
feat(model): add beam search generation
```

## Documentation

Update documentation for any changes:
1. Update docstrings
2. Update README if needed
3. Add/update guides in /docs
4. Build docs locally to verify

## Code Review Guidelines

- Keep PRs focused and small
- Add tests for new features
- Update documentation
- Follow existing code style
- Resolve all CI checks

## Release Process

1. Update version in setup.py
2. Update CHANGELOG.md
3. Create release PR
4. Tag release on GitHub
5. Deploy to PyPI