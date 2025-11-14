# Contributing to CAD Fusion Lab

Thank you for considering contributing to CAD Fusion Lab! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Feature Requests](#feature-requests)

## Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow:

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

### Prerequisites

- Python 3.9+
- Git
- (Optional) CUDA-compatible GPU for training
- (Optional) Docker for containerized development

### Development Environment Setup

1. **Fork and clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/cad-fusion-lab.git
cd cad-fusion-lab
```

2. **Set up development environment**

```bash
# Quick setup
make init-dev

# Or manual setup
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
```

3. **Copy environment configuration**

```bash
cp .env.example .env
# Edit .env with your configuration
```

4. **Verify installation**

```bash
make test
make lint
```

## Development Workflow

### Branch Strategy

- `main` - Production-ready code
- `develop` - Integration branch for features
- `feature/*` - New features
- `bugfix/*` - Bug fixes
- `hotfix/*` - Urgent production fixes

### Creating a Branch

```bash
git checkout -b feature/your-feature-name
```

### Making Changes

1. **Write code** following our [Code Standards](#code-standards)
2. **Add tests** for new functionality
3. **Update documentation** as needed
4. **Run quality checks**

```bash
make format  # Format code
make lint    # Run linters
make test    # Run tests
```

5. **Commit changes** using conventional commits

```bash
git add .
git commit -m "feat: add new CAD export format"
```

### Conventional Commit Messages

We use conventional commits for clear and automated changelog generation:

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks
- `perf:` - Performance improvements

Examples:
```
feat: add IGES export support
fix: resolve memory leak in CAD decoder
docs: update installation instructions
test: add integration tests for API endpoints
```

## Code Standards

### Python Style Guide

We follow PEP 8 with these specific guidelines:

- **Line length**: 100 characters
- **Formatting**: Use Black for code formatting
- **Import sorting**: Use isort with black profile
- **Type hints**: Required for public APIs
- **Docstrings**: Google style for all public functions/classes

### Example

```python
from typing import List, Optional

def generate_cad_sequence(
    text: str,
    max_length: int = 512,
    temperature: float = 0.8
) -> List[int]:
    """
    Generate CAD token sequence from text description.

    Args:
        text: Natural language description of CAD model
        max_length: Maximum sequence length
        temperature: Sampling temperature (0.1-2.0)

    Returns:
        List of CAD operation tokens

    Raises:
        ValueError: If temperature is out of range
    """
    if not 0.1 <= temperature <= 2.0:
        raise ValueError("Temperature must be between 0.1 and 2.0")

    # Implementation...
    return []
```

### Code Organization

- **Modularity**: One class/function per logical unit
- **Single Responsibility**: Each module should have one clear purpose
- **DRY**: Don't Repeat Yourself - extract common logic
- **SOLID**: Follow SOLID principles where applicable

### Security Guidelines

- **Never commit secrets**: Use environment variables
- **Input validation**: Validate all user inputs
- **SQL injection**: Use parameterized queries
- **XSS prevention**: Sanitize text outputs
- **Authentication**: Use provided auth framework

## Testing

### Test Structure

```
tests/
├── unit/              # Unit tests
├── integration/       # Integration tests
├── fixtures/          # Test fixtures and data
└── conftest.py       # Pytest configuration
```

### Writing Tests

```python
import pytest
from src.models.text_to_cad import TextToCADModel

def test_model_initialization():
    """Test model initializes correctly."""
    model = TextToCADModel(vocab_size=1000, offline_mode=True)
    assert model.vocab_size == 1000

@pytest.mark.integration
def test_full_pipeline():
    """Test complete generation pipeline."""
    # Integration test code...
    pass
```

### Running Tests

```bash
# All tests
make test

# Fast tests (no coverage)
make test-fast

# Integration tests only
make test-integration

# Specific test file
pytest tests/test_models.py -v

# Specific test
pytest tests/test_models.py::test_model_initialization -v
```

### Test Coverage

- Aim for >80% code coverage
- All public APIs must have tests
- Edge cases and error conditions must be tested

## Pull Request Process

### Before Submitting

1. **Update your branch**

```bash
git checkout develop
git pull origin develop
git checkout your-branch
git rebase develop
```

2. **Run all checks**

```bash
make check  # Runs format, lint, and test
```

3. **Update documentation**
   - Update README if adding features
   - Update docstrings
   - Add examples if appropriate

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests pass locally
- [ ] No new warnings

## Related Issues
Closes #123
```

### Review Process

1. **Automated checks** must pass (CI/CD)
2. **Code review** by at least one maintainer
3. **Testing** verification
4. **Documentation** review
5. **Approval** and merge

### After Merge

- Delete your feature branch
- Pull latest changes to local develop
- Pat yourself on the back!

## Reporting Bugs

### Before Reporting

1. **Check existing issues** - May already be reported
2. **Try latest version** - Bug may be fixed
3. **Reproduce** - Confirm it's reproducible

### Bug Report Template

```markdown
**Describe the bug**
Clear description of the bug

**To Reproduce**
1. Step 1
2. Step 2
3. Error occurs

**Expected behavior**
What should happen

**Environment**
- OS: [e.g., Ubuntu 22.04]
- Python: [e.g., 3.10]
- Version: [e.g., 1.0.0]

**Additional context**
Logs, screenshots, etc.
```

## Feature Requests

We welcome feature requests! Please:

1. **Check existing requests** first
2. **Describe the problem** your feature would solve
3. **Propose a solution** if you have one
4. **Consider alternatives** you've thought about

### Feature Request Template

```markdown
**Is your feature request related to a problem?**
Description of the problem

**Describe the solution you'd like**
Clear description of desired feature

**Describe alternatives you've considered**
Other solutions you've thought about

**Additional context**
Mockups, examples, etc.
```

## Questions?

- **Documentation**: Check our [README](README.md) and docs
- **Discussions**: Use GitHub Discussions for questions
- **Chat**: Join our community chat (if available)
- **Email**: Contact maintainers at [email]

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in the project

Thank you for contributing to CAD Fusion Lab! 🚀
