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

### Update HF Model Personality dataset

The HuggingFace dataset [rdnfn/ff-model-personality](https://huggingface.co/datasets/rdnfn/ff-model-personality) is updated via a GitHub action in the [Feedback Forensics repo](https://github.com/rdnfn/feedback-forensics). By configuring the action script, you can add new models to the dataset - e.g. following a new model release. All models available via openrouter are supported. The following steps are required to add a new model:

1. Add the new model to the `.github/workflows/update-model-comparison.yml` config file under the `env.MODELS` parameter. Note that models need to be separated by a *space* (not a comma).
2. Merge the change onto the `main` branch (i.e. by opening and merging a PR). Once a new FF version is released, the new model will be added to the dataset. Alternatively, you can request an FF maintainer to manually trigger a dataset update ahead of the next release.