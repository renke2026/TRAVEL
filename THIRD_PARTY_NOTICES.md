# Third-party notices

Original TRAVEL software is licensed under MIT. The following resources retain their own attribution and terms.

## SynthesizeMe

The transport-task adaptation in `baselines/prompt_baselines.py`, `full_synthesizeme.py`, `reasoned_readout.py`, and `budget_synthesizeme.py` follows the official [SALT-NLP/SynthesizeMe](https://github.com/SALT-NLP/SynthesizeMe) implementation, including the source revision identified in the code. Copyright (c) 2025 Michael Ryan. The original MIT license is preserved verbatim at [licenses/SynthesizeMe-LICENSE](licenses/SynthesizeMe-LICENSE).

## Related implementations and methods

- [Persona Vectors](https://github.com/safety-research/persona_vectors) and [PERSONA](https://github.com/xcfcode/persona) inform contrastive activation extraction and intervention. Their repositories are not bundled here.
- The [PBEL method](https://doi.org/10.1287/trsc.2025.0330) is implemented as a task adaptation under this study's common model and data budget.
- Travel-preference constructs draw on [Vredin Johansson, Heldt, and Johansson (2006)](https://doi.org/10.1016/j.tra.2005.09.001). The released synthetic prompts/questions are this study's adaptation, rather than the original questionnaire.

## Data, models, and libraries

Train and Swissmetro originate from the public sources documented in [docs/DATA.md](docs/DATA.md). Original survey files are obtained from those sources and are not included in the release. Anonymous identifiers/labels in the compact evaluation artifacts retain their relation to those sources.

Qwen and Llama weights are obtained separately under their publishers' terms. This repository's MIT license does not replace model licenses or data-use terms. Python dependencies are installed separately and retain their respective licenses.
