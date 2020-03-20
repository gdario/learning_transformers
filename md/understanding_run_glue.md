Understanding `run_glue.py`
===========================

`WEIGHTS_NAME` is a constant string containing `pytorch_model.bin`.
There might be more information
[here](https://huggingface.co/transformers/serialization.html?highlight=weights_name).

Modules called in the script
----------------------------

The script calls a number of modules. Some of them are documented
(processors), some are not (the first two).

``` {.example}
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from transformers import glue_convert_examples_to_features as convert_examples_to_features
```

`glue_compute_metrics` is defined in `__init__.py` and it returns, given
a GLUE task, the appropriate metrics. `glue_processors` and
`glue_output_modes` are defined in `glue.py` and are constants
associating the correct processors to each task and specifying whether
the task is a classification or a regression one, respectively.
