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

## Docs

See https://feedback-forensics.readthedocs.io.

## Online usage

See our [online platform](https://app.feedbackforensics.com) to track personality traits in popular models and datasets. No local installation required.

## Local usage

To track personality traits in your own datasets and models, install Feedback Forensics locally.

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

**Next steps**

See the [getting started guides in the docs](https://feedback-forensics.readthedocs.io/en/dev-docs-update/guide/index.html) to analyse your own feedback datasets and models.

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

## Citation

If you find Feedback Forensics useful in your research, please consider citing the project:

```bibtex
@software{feedbackforensics,
  author = {Findeis, Arduin and Kaufmann, Timo and H{\"u}llermeier, Eyke and Mullins, Robert},
  title = {Feedback Forensics: An open-source toolkit to measure AI personality changes},
  url = {https://github.com/rdnfn/feedback-forensics},
  year = {2025}
}
```

## License

[Apache 2.0](LICENSE)