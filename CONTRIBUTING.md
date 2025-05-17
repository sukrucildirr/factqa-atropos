# Contributing to FactQA Atropos

Thank you for considering contributing to the FactQA Atropos project! This document provides guidelines and instructions for contributing to make the process smooth and effective for everyone involved.

## Code of Conduct

By participating in this project, you agree to abide by the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). Please report unacceptable behavior to the project maintainers.

## How to Contribute

### Reporting Bugs

If you find a bug in the project, please create an issue on GitHub with the following information:

- A clear, descriptive title
- A detailed description of the issue, including steps to reproduce
- Your environment details (OS, Python version, etc.)
- Any relevant logs or error messages
- Possible solutions or workarounds if you have any

### Suggesting Enhancements

We welcome suggestions for enhancements! Please create an issue on GitHub with:

- A clear, descriptive title
- A detailed description of the proposed enhancement
- Any relevant examples, mockups, or use cases
- Potential implementation approaches if you have ideas

### Pull Requests

1. Fork the repository
2. Create a new branch for your feature or bugfix (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests to ensure your changes don't break existing functionality
5. Commit your changes with clear, descriptive commit messages
6. Push to your branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request against the main repository

#### Pull Request Guidelines

- Follow the coding style of the project
- Include tests for new features or bug fixes
- Update documentation as needed
- Keep pull requests focused on a single change
- Link any relevant issues in the pull request description

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/factqa-atropos.git
   cd factqa-atropos
   ```

2. Install in development mode:
   ```bash
   pip install -e .
   ```

3. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt  # if available
   ```

## Testing

Run tests using pytest:

```bash
pytest
```

## Documentation

Please update documentation when making changes:

- Update README.md for user-facing changes
- Update docstrings for code changes
- Add examples for new features

## Versioning

We use [Semantic Versioning](https://semver.org/) for releases:

- MAJOR version for incompatible API changes
- MINOR version for backwards-compatible functionality additions
- PATCH version for backwards-compatible bug fixes

## License

By contributing to this project, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).

## Questions?

If you have any questions about contributing, please open an issue or contact the project maintainers.

Thank you for your contributions!
