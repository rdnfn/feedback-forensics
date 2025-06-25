## Contributing 🌻

### Setup

If you want to contribute to Feedback Forensics, there are two options to set up the development environment:

#### Option 1: Standard development setup

1. Clone this repository
2. Install the package with development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

#### Option 2: Development container

For a consistent development environment, this repository includes a VS Code dev container configuration:

1. Install the [Remote - Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
2. Open the repository in VS Code
3. Click "Reopen in Container" when prompted

### Running test cases

To run the tests for the package, run:

```bash
pytest ./src
```

### Creating a PR

First create a PR to the `staging` branch, from there the work will then be merged with the main branch. A merge (and push) in the `staging` branch will allow you to view the staged online version of Feedback Forensics app at https://rdnfn-ff-dev.hf.space.


### Creating a new release

Ensure that the current branch is up-to-date with main, and then bump the version (using `patch`, `minor`, or `major`):
```bash
bump-my-version bump patch
```

Then on the GitHub website create a new release named after the new version (e.g. "v0.1.2"). As part of this release in the GitHub interface, create a new tag with the updated version. This release will trigger a GitHub action to build and upload the PyPI package.

### Creating docs locally

If you want to test and compile the docs locally, run:
```bash
jupyter-book build docs/
```