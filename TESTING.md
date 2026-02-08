# Testing with tox

This project uses tox for testing across multiple Python versions.

## Installation

```bash
pip install tox
```

## Usage

### Test on all Python versions
```bash
tox
```

### Test on specific Python version
```bash
tox -e py38   # Python 3.8
tox -e py39   # Python 3.9
tox -e py310  # Python 3.10
tox -e py311  # Python 3.11
tox -e py312  # Python 3.12
tox -e py313  # Python 3.13
tox -e py314  # Python 3.14
```

### Test on multiple versions
```bash
tox -e py310,py311,py312
```

### Recreate environments
If you've updated dependencies:
```bash
tox -r
```

### Run tests in parallel
```bash
tox -p auto
```

### Lint code
```bash
tox -e lint
```

### Format code
```bash
tox -e format
```

## Configuration

The tox configuration is in `tox.ini`. Each Python version has specific dependency versions that are compatible with that Python version:

- **Python 3.8-3.9**: numpy<2.0, older versions of dependencies
- **Python 3.10-3.11**: numpy 1.23-1.24, mid-range dependency versions  
- **Python 3.12**: numpy>=1.26, current versions
- **Python 3.13-3.14**: numpy>=2.0, latest versions

## Requirements

You need to have the corresponding Python versions installed on your system. You can use:
- **pyenv**: For managing multiple Python versions
- **deadsnakes PPA** (Ubuntu/Debian): `sudo add-apt-repository ppa:deadsnakes/ppa`
- **Official Python downloads**: https://www.python.org/downloads/

## CI/CD

For automated testing in CI/CD, consider using:
- GitHub Actions (see example in `.github/workflows/tests.yml`)
- GitLab CI
- Jenkins
- CircleCI

## Notes

- Python 3.13 and 3.14 may require beta/RC releases as they are not yet stable
- Some dependencies may not support the latest Python versions immediately
- Test data is cached in `tests/test_data/` and shared across all test runs

