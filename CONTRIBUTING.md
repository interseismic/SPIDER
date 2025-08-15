# Contributing to SPIDER

Thank you for your interest in contributing to SPIDER! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Types of Contributions

We welcome various types of contributions:

- **Bug Reports**: Report bugs and issues
- **Feature Requests**: Suggest new features
- **Code Contributions**: Submit pull requests
- **Documentation**: Improve docs and examples
- **Testing**: Add tests or report test results
- **Examples**: Create new usage examples

### Before You Start

1. **Check existing issues** to avoid duplicates
2. **Read the documentation** to understand the codebase
3. **Set up development environment** (see below)
4. **Follow coding standards** (see below)

## 🛠️ Development Setup

### Prerequisites

- Python 3.8+
- Git
- CUDA-compatible GPU (recommended)
- PyTorch with CUDA support

### Local Development

```bash
# Fork and clone the repository
git clone https://github.com/your-username/SPIDER.git
cd SPIDER

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install
```

### Testing

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=spider

# Run specific test file
python -m pytest tests/test_core.py

# Run with verbose output
python -m pytest tests/ -v
```

## 📝 Coding Standards

### Python Style

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

- **Line length**: 88 characters (Black default)
- **Docstrings**: Google style
- **Type hints**: Required for public functions
- **Imports**: Organized with `isort`

### Code Formatting

We use automated tools for code formatting:

```bash
# Format code with Black
black spider/ tests/

# Sort imports with isort
isort spider/ tests/

# Check formatting
black --check spider/ tests/
isort --check-only spider/ tests/
```

### Pre-commit Hooks

The repository includes pre-commit hooks that automatically:

- Format code with Black
- Sort imports with isort
- Check for common issues
- Run basic tests

Install with:
```bash
pre-commit install
```

## 🐛 Bug Reports

### Before Reporting

1. **Search existing issues** for similar problems
2. **Check the documentation** for solutions
3. **Test with latest version** from main branch
4. **Reproduce the issue** with minimal example

### Bug Report Template

```markdown
**Bug Description**
Brief description of the issue.

**Steps to Reproduce**
1. Step 1
2. Step 2
3. Step 3

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened.

**Environment**
- OS: [e.g., Ubuntu 20.04]
- Python: [e.g., 3.9.7]
- PyTorch: [e.g., 1.12.1]
- CUDA: [e.g., 11.6]
- SPIDER version: [e.g., 0.1.0]

**Additional Information**
- Error messages/logs
- Screenshots (if applicable)
- Configuration files (anonymized)
```

## 💡 Feature Requests

### Before Requesting

1. **Check existing issues** for similar requests
2. **Consider the scope** and complexity
3. **Think about implementation** approach
4. **Consider backward compatibility**

### Feature Request Template

```markdown
**Feature Description**
Brief description of the requested feature.

**Use Case**
Why this feature would be useful.

**Proposed Implementation**
How you think it could be implemented (optional).

**Alternatives Considered**
Other approaches you've considered (optional).

**Additional Context**
Any other relevant information.
```

## 🔧 Code Contributions

### Pull Request Process

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
   - Follow coding standards
   - Add tests for new functionality
   - Update documentation
4. **Test your changes**
   ```bash
   python -m pytest tests/
   black --check spider/ tests/
   isort --check-only spider/ tests/
   ```
5. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add feature: brief description"
   ```
6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Create a pull request**

### Pull Request Template

```markdown
**Description**
Brief description of changes.

**Type of Change**
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Test addition
- [ ] Other (please describe)

**Testing**
- [ ] Tests pass locally
- [ ] Added tests for new functionality
- [ ] Updated existing tests

**Documentation**
- [ ] Updated docstrings
- [ ] Updated README/docs
- [ ] Added examples

**Breaking Changes**
- [ ] Yes (describe changes)
- [ ] No

**Additional Notes**
Any other information.
```

### Code Review Guidelines

#### For Contributors

- **Respond to feedback** promptly
- **Address all comments** before merging
- **Keep commits focused** and well-described
- **Test thoroughly** before requesting review

#### For Reviewers

- **Be constructive** and respectful
- **Focus on code quality** and functionality
- **Check for security issues**
- **Ensure tests are adequate**
- **Verify documentation updates**

## 📚 Documentation Contributions

### Types of Documentation

- **Code docstrings**: Function and class documentation
- **README updates**: Main project documentation
- **Example scripts**: Usage examples
- **Tutorial notebooks**: Step-by-step guides
- **API documentation**: Detailed API reference

### Documentation Standards

- **Clear and concise** writing
- **Code examples** for all functions
- **Type hints** in docstrings
- **Cross-references** to related functions
- **Version compatibility** notes

### Building Documentation

```bash
# Install documentation dependencies
pip install sphinx sphinx-rtd-theme

# Build documentation
cd docs/
make html

# View locally
open _build/html/index.html
```

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── test_core/           # Core functionality tests
├── test_io/            # I/O functionality tests
├── test_analysis/      # Analysis functionality tests
├── test_utils/         # Utility function tests
└── conftest.py         # Test configuration
```

### Writing Tests

- **Test one thing** per test function
- **Use descriptive names** for test functions
- **Include edge cases** and error conditions
- **Mock external dependencies** when appropriate
- **Use fixtures** for common setup

### Test Example

```python
import pytest
import numpy as np
from spider.core.data import prepare_input_dfs

def test_prepare_input_dfs_valid_data():
    """Test data preparation with valid input."""
    # Arrange
    params = {
        "dtime_file": "tests/data/valid_dtimes.csv",
        "catalog_infile": "tests/data/valid_events.csv"
    }
    
    # Act
    stations, dtimes, origins = prepare_input_dfs(params)
    
    # Assert
    assert stations is not None
    assert dtimes is not None
    assert origins is not None
    assert len(origins) > 0

def test_prepare_input_dfs_missing_file():
    """Test data preparation with missing file."""
    # Arrange
    params = {
        "dtime_file": "nonexistent_file.csv",
        "catalog_infile": "tests/data/valid_events.csv"
    }
    
    # Act & Assert
    with pytest.raises(FileNotFoundError):
        prepare_input_dfs(params)
```

## 🚀 Release Process

### Version Numbers

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

- [ ] **Update version** in `setup.py`
- [ ] **Update changelog** with new features/fixes
- [ ] **Run full test suite** and ensure all tests pass
- [ ] **Update documentation** if needed
- [ ] **Create release tag**
- [ ] **Build and upload** to PyPI (if applicable)

## 📞 Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Pull Requests**: Code contributions and reviews

### Before Asking for Help

1. **Check the documentation**
2. **Search existing issues**
3. **Try to reproduce** the issue
4. **Prepare a minimal example**

## 📄 License

By contributing to SPIDER, you agree that your contributions will be licensed under the same license as the project (MIT License).

## 🙏 Recognition

Contributors will be recognized in:

- **Contributors list** on GitHub
- **Release notes** for significant contributions
- **Documentation** for major features
- **Academic publications** if applicable

---

Thank you for contributing to SPIDER! Your contributions help make earthquake location more accurate and accessible to the scientific community.
