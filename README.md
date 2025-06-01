<p align="center">
  <a href="https://app.feedbackforensics.com/">
  <img src="src/feedback_forensics/assets/feedback_forensics_logo.png" alt="Feebdack Forensics Logo" width="330px"></a>
  <br>
  <a href="https://app.feedbackforensics.com/">
  <img src="docs/img/demo_v4.gif" alt="" width="650px"></a>
  <br>
  <a href="https://app.feedbackforensics.com/">
  <img src="docs/img/button_demo_v2.png" alt="Run demo" width="170px"></a>
  <br>
  <a href="https://pypi.org/project/feedback-forensics/">
  <img alt="PyPI" src="https://img.shields.io/pypi/v/feedback-forensics?logo=python&logoColor=f59e0d&labelColor=black&color=52525b"></a>
  <a href="https://github.com/rdnfn/feedback-forensics/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/pypi/l/feedback-forensics?labelColor=black&color=52525b"></a>
  <a href="https://feedback-forensics.readthedocs.io/en/latest/">
  <img alt="Read the Docs" src="https://img.shields.io/readthedocs/feedback-forensics?labelColor=black&logo=readthedocs&logoColor=white"></a>
  <a href="https://github.com/rdnfn/feedback-forensics/deployments/pypi">
  <img alt="GitHub deployments" src="https://img.shields.io/github/deployments/rdnfn/feedback-forensics/pypi?label=package%20build&labelColor=black&logo=github&logoColor=white"></a>
</p>


**Feedback Forensics is an open-source toolkit to measure AI personality changes**. Beyond raw capabilities, *model personality traits*, such as tone and sycophancy, also matter to users. Feedback Forensics can help you track *(1) personality changes encouraged by your human (or AI) feedback datasets* ([tutorial](https://feedback-forensics.readthedocs.io/en/latest/guide/feedback.html)), and *(2) personality traits exhibited by your AI models* ([tutorial](https://feedback-forensics.readthedocs.io/en/latest/guide/models.html)). Feedback Forensics includes a *Python API*, an *annotation CLI*, and a *[Gradio](https://www.gradio.app/) visualisation app*. We also provide a corresponding [online platform](https://app.feedbackforensics.com) tracking personality traits in popular models and datasets.


| *Use-case 1:*<br>Finding personality changes encouraged by feedback data| *Use-case 2:*<br>Measuring personality changes across models|
|:---:|:---:|
|*What personality traits is Chatbot Arena encouraging?*|*What personality traits changed between Llama 3 and Llama 4?*|
|<img src="docs/img/example_feedback_v1.png" alt="example_feedback" width="350px">|<img src="docs/img/example_models_v1.png" alt="example_models" width="350px">|
<a href="https://app.feedbackforensics.com?data=chatbot_arena"><img src="docs/img/button_demo_v2.png" alt="Run demo" width="110px"></a>  <a href="https://feedback-forensics.readthedocs.io/en/latest/guide/feedback.html"><img src="docs/img/button_tutorial.png" alt="Open tutorial" width="110px"></a>|<a href="https://app.feedbackforensics.com/?data=model_comparison&ann_cols=model_metallamallama370binstruct,model_metallamallama4maverick"><img src="docs/img/button_demo_v2.png" alt="Run demo" width="110px"></a>  <a href="https://feedback-forensics.readthedocs.io/en/latest/guide/models.html"><img src="docs/img/button_tutorial.png" alt="Open tutorial" width="110px"></a>|


## Local usage

### Installation

```sh
pip install feedback-forensics
```

### Getting started

To start the app locally, run the following command in your terminal:

```sh
feedback-forensics -d data/output/example/annotated_pairs.json
```

This will start the Gradio interface on localhost port 7860 (e.g. http://localhost:7860).

> [!NOTE]
> The online results are currently not available when running locally.

### Investigating your own dataset

To investigate your own dataset, you first need to annotate your data with principle-following annotators. Using such annotators first requires setting API keys in `secrets.toml` file, as [described here](https://github.com/rdnfn/icai?tab=readme-ov-file#installation). Then, to annotate your data, run:

```shell
ff-annotate --datapath="data/input/example.csv"
```

Replace `example.csv` with your own dataset, ensuring it complies with the ICAI standard data format (as described [here](https://github.com/rdnfn/icai?tab=readme-ov-file#run-experiment-with-your-own-data), i.e. containing columns `text_a`, `text_b`, and `preferred_text`). For comparability, this will annotate your data using the *Feedback Forensics standard principles* rather than generating new ones. These standard principles are used to created the online interface results (shown as the *implicit objectives*).

Once the experiment is completed, run the following command (also shown at end of ICAI experiment terminal output):

```shell
feedback-forensics -d /path/to/your/icai_results/070_annotations_train_ap.json
```

This command will again open up the feedback forensics app on localhost port 7860, now including the local results on your own dataset.

**Additional options.** Alternatively to `ff-annotate` which uses ICAI under the hood, you can also use ICAI directly to get access to all configuration parameters. The equivalent command to the `ff-annotate` above is:

```shell
icai-exp data_path="data/input/example.csv" s0_added_standard_principles_to_test="[v2]" annotator.skip=true s0_skip_principle_generation=true
```

The last two arguments (`annotator.skip` and `s0_skip_principle_generation`) reduce experiment cost by skipping parts not necessary for feedback forensics visualisation. Set `s0_skip_principle_generation=false` to additionally generate new principles beyond the standard set.

### Python interface

Feedback Forensics can also be used to interpret annotator data within Python. Below is a minimal example:

```python
import feedback_forensics as ff

# load dataset from AnnotatedPairs json file produced by ICAI package
dataset = ff.DatasetHandler()
dataset.add_data_from_path("data/output/example/annotated_pairs.json")

overall_metrics = dataset.get_overall_metrics()
annotator_metrics = dataset.get_annotator_metrics()
```

## Limitations

Feedback Forensics relies on AI annotators (LLM-as-a-Judge) to detect implicit objectives in feedback data. Though such annotators have been shown correlate with human judgements on many [tasks](https://github.com/tatsu-lab/alpaca_eval/tree/main/src/alpaca_eval/evaluators_configs), they also have well-known limitations: they are often susceptible to small input changes and can exhibit [various](https://arxiv.org/abs/2405.01724) [biases](https://arxiv.org/abs/2306.05685) (as do [human annotators](https://arxiv.org/abs/2309.16349)). As such, *Feedback Forensics results should be taken as an indication for further investigation rather than a definitive final judgement of the data*. In general, results based on more samples are less susceptible to noise introduced by AI annotators – and thus may be considered more reliable.

## Development

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
```
bump-my-version bump patch
```

Then on the GitHub website create a new release named after the new version (e.g. "v0.1.2"). As part of this release in the GitHub interface, create a new tag with the updated version. This release will trigger a GitHub action to build and upload the PyPI package.


## License

[Apache 2.0](LICENSE)
