<p align="center">
  <a href="https://rdnfn-feedback-forensics.hf.space/">
  <img src="src/forensics/assets/feedback_forensics_logo.png" alt="Feebdack Forensics Logo" width="330px">
  </a>
  <br>
  <a href="https://rdnfn-feedback-forensics.hf.space/">
  <img src="docs/img/demo_v3.gif" alt="" width="650px">
  </a>
  <br>
  <a href="https://rdnfn-feedback-forensics.hf.space/">
  <img src="docs/img/run_demo_button.png" alt="Run demo" width="150px">
  </a>
  <br>
  <a href="https://pypi.org/project/forensics/">
  <img alt="PyPI" src="https://img.shields.io/pypi/v/forensics?logo=python&logoColor=f59e0d&labelColor=black&color=52525b">
  </a>
</p>


**Feedback Forensics is a tool to investigate pairwise feedback data used for AI training and evaluation:** when used for training, what is the data teaching our models? When used for evaluation, towards what kind of models is the feedback leading us? Is this feedback asking for more lists or more ethically considerate responses? Feedback Forensics enables answering these kind of questions, building on the [Inverse Constitutional AI](https://github.com/rdnfn/icai) (ICAI) pipeline to automatically detect and measure the *implicit objectives* of annotations. Feedback Forensics is an [open-source](https://github.com/rdnfn/feedback-forensics/blob/main/LICENSE) [Gradio](https://www.gradio.app/) app that can be used both [online](https://rdnfn-feedback-forensics.hf.space) and [locally](#local-installation).

> *"Investigate your pairwise feedback data"* 🕵🏻‍♂️💬




| Example use-cases | Screenshots |
|----------|-------------|
| **🗂️Visualizing dataset differences**<br>*How does Chatbot Arena differ from Anthropic Helpful data?*<br><a href="https://rdnfn-feedback-forensics.hf.space?data=chatbot_arena,anthropic_helpful"><img src="docs/img/run_demo_button.png" alt="run demo" width="100px"></a>| <img src="docs/img/example01_v4.png" alt="example_screenshot_01" width="1000px"> |
| **🧑‍🎨🧑‍💼 Finding preference differences between task domains**<br>*How do preferences differ across writing tasks?*<br><a href="https://rdnfn-feedback-forensics.hf.space?data=chatbot_arena&col=narrower_category&col_vals=songwriting_prompts,resume_and_cover_letter_writing,professional_email_communication,creative_writing_prompts"><img src="docs/img/run_demo_button.png" alt="run demo" width="100px"></a>  | <img src="docs/img/example02_v4.png" alt="example_screenshot_02" width="1000px"> |
| **🤖 Discovering model strengths**<br>*How is GPT-4o different to other models?*<br><a href="https://rdnfn-feedback-forensics.hf.space?data=chatbot_arena&col=winner_model&col_vals=gpt4o20240513,claude35sonnet20240620,gemini15proapi0514,mistrallarge2407,deepseekv2api0628"><img src="docs/img/run_demo_button.png" alt="run demo" width="100px"></a>|<img src="docs/img/example03_v4.png" alt="example_screenshot_03" width="1000px">|


## Local usage

### Installation

```sh
pip install feedback-forensics
```

### Getting started

To start the app locally, run the following command in your terminal

```sh
feedback-forensics
```

### Investigating your own dataset

First run an [ICAI](https://github.com/rdnfn/icai) experiment on your dataset. For example, you can evaluate the same implicit objectives as in the online interface results using ICAI standard principles. First install ICAI as described [here](https://github.com/rdnfn/icai?tab=readme-ov-file#installation), including setting up relevant API secrets. Then, run:

```
icai-exp data_path="data/processed/example/example.csv" s0_added_standard_principles_to_test="[v2]" annotator.skip=true s0_skip_principle_generation=true
```

The last two arguments (`annotator.skip` and `s0_skip_principle_generation`) reduce experiment cost by skipping testing a final constitutional annotator (not necessary for feedback forensics visualiation), and skipping the generation and testing of new (non-standard) principles. Replace `example.csv` with your own dataset, ensuring it complies with the ICAI standard data format (as described [here](https://github.com/rdnfn/icai?tab=readme-ov-file#run-experiment-with-your-own-data)).

## License

[Apache 2.0](LICENSE)