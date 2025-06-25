# Installation

```{seealso}
If you want to skip local installation for now, see our online platform at [app.feedbackforensics.com](https://app.feedbackforensics.com/) for ready-to-view personality results on popular datasets and models.
```

## Install from PyPI

This is the **recommended** installation method.

```bash
pip install feedback-forensics
```

## Install from Source

```bash
git clone https://github.com/rdnfn/feedback-forensics.git
cd feedback-forensics
pip install -e .
```

## API Configuration (secrets.toml)

To use Feedback Forensics for annotating your own datasets, you need to configure API keys for the AI models used in the annotation process. This can either be done through a `secrets.toml` file or by setting environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `OPENROUTER_API_KEY`) whenever running annotations.

### Setting up secrets.toml

1. Create a `secrets.toml` file in your working directory (or wherever you plan to run the `ff-annotate` command)

2. Add your API keys to the file in the following format:

```toml
OPENAI_API_KEY="your-openai-api-key-here"
ANTHROPIC_API_KEY="your-anthropic-api-key-here"
OPENROUTER_API_KEY="your-openrouter-api-key-here"
```

Feedback Forensics currently only supports three API providers: `OpenAI`, `Anthropic`, and `OpenRouter`. You only need to include keys for the APIs you plan to use.


```{warning}
Do not commit the `secrets.toml` file. It is recommended to add this file to your project's `.gitignore` file to avoid this.
```

## Next Steps

- [Getting Started Guides](guide/index.md)