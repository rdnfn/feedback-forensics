# API Reference 🔌

## Python API

Feedback Forensics can be used to interpret annotator data within Python. Below is a minimal example:

```python
import feedback_forensics as ff

# load dataset from AnnotatedPairs json file produced by ICAI package
dataset = ff.DatasetHandler()
dataset.add_data_from_path("data/output/example/annotated_pairs.json")

overall_metrics = dataset.get_overall_metrics()
annotator_metrics = dataset.get_annotator_metrics()
```

## Command Line Interface

### feedback-forensics
Main CLI command for launching the analysis interface.

```bash
feedback-forensics [OPTIONS]

options:
  -h, --help            show help message and exit
  --datapath, -d DATAPATH
                        Path to dataset
```

### ff-annotate
CLI command for annotating datasets with ICAI.

```bash
ff-annotate [OPTIONS]

options:
  -h, --help            show help message and exit
  --datapath, -d DATAPATH
                        Path to dataset CSV file with columns text_a, text_b, and preferred_text
  --principles-version, -p PRINCIPLES_VERSION
                        Version of standard principles to test (default: v2)
```

