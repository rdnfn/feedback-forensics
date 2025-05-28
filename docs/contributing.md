# Contributing 📮

## Getting Started

### Development Setup

1. Clone the repository:
```bash
git clone https://github.com/rdnfn/feedback-forensics.git
cd feedback-forensics
```

2. Install development dependencies:
```bash
pip install -e ".[dev]"
```

### Development Container (Optional)

Use dev container for consistent environment:

1. Install 'Remote - Containers' extension
2. Open repository in VS Code (or Cursor)
3. Click "Reopen in Container" when prompted

## Development Workflow

### Making Changes

1. Create a feature branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes
3. Run tests:
```bash
pytest ./src
```

4. Format code:
```bash
black src/
```

5. Commit and push changes

### Pull Request Process

1. Create PR to `staging` branch first
2. Ensure all tests pass
3. Request review from maintainers
4. Address feedback
5. Merge to `staging`, then to `main`


## Release Process

### Version Bumping
```bash
bump-my-version bump patch  # or minor/major
```

### Creating Releases
1. Ensure `main` branch is up to date
2. Create GitHub release with version tag
3. GitHub Actions will build and upload to PyPI


## Documentation Development

### Building Documentation
```bash
# Install Jupyter Book
pip install -e ".[docs]"

# Build documentation
jupyter-book build docs/

# View locally
open docs/_build/html/index.html
```

