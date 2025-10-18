# Contributing to GasTwinFormer

Thank you for your interest in contributing to GasTwinFormer! We welcome contributions from the community.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue on GitHub with:
- A clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment information (OS, Python version, PyTorch version, etc.)

### Suggesting Enhancements

Enhancement suggestions are welcome! Please create an issue with:
- A clear description of the enhancement
- Motivation and use cases
- Potential implementation approach

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** following our coding style
3. **Add tests** if applicable
4. **Update documentation** if needed
5. **Ensure all tests pass**
6. **Submit a pull request**

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/gastwinformer.git
cd gastwinformer

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks (optional but recommended)
pip install pre-commit
pre-commit install
```

## Coding Style

- Follow PEP 8 style guide
- Use type hints where appropriate
- Write clear docstrings (Google style)
- Keep functions focused and modular

## Testing

```bash
# Run tests
pytest tests/

# Check code style
flake8 gastwinformer/
black --check gastwinformer/
isort --check gastwinformer/
```

## Commit Messages

- Use clear, descriptive commit messages
- Start with a verb in present tense (e.g., "Add", "Fix", "Update")
- Reference issues when applicable (e.g., "Fix #123")

## Questions?

Feel free to open an issue for any questions or concerns!

