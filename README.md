# TRAVEL

**Travel-preference Representation via Activation VEctors in Language models**

Research code for representing travel preferences with activation vectors and evaluating their effects on language-model travel choices. The five dimensions are **environment, safety, comfort, convenience, and flexibility**.

Each vector is the difference between mean hidden states under high- and low-concern conditions. Interventions use `h += alpha * v` with the original vector magnitude. TRAVEL freezes the language-model weights; the Swissmetro application estimates shared statistical parameters and personal mixture weights from observed choices.

This repository accompanies the submission manuscript by Xiangyu He, Ziyuan Gu, and Zhiyuan Liu. It contains the experiment source code, fixed materials, small extracted vectors, fitted application settings, and compact numerical outputs needed to inspect the principal results.

## Quick start: reproduce the numerical results

This path needs Python 3.11 and NumPy. It requires no GPU, model download, API key, or third-party dataset download.

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-analysis.txt
python scripts/check_release.py
python scripts/reproduce_tables.py
```

The second command recomputes the selected-layer behavioral results, Train evaluation and response ordering, Swissmetro prediction and perturbation metrics, and cross-model gains. CSV outputs are written to `outputs/reproduced/`.

Expected Swissmetro method-selection results:

| Method | Accuracy | NLL |
| --- | ---: | ---: |
| TRAVEL | 71.76% | 0.606 |
| LoRA-SFT | 73.61% | 0.565 |

These values concern the 72-person method-selection group, with three evaluation choices per person. Perturbation results use a separate 35-person group. `source_data/` contains generated model outputs and evaluation metadata; the original survey files are obtained separately.

## Repository layout

| Location | Contents |
| --- | --- |
| `scripts/` | Public entry points for checks, data preparation, numerical reproduction, and model experiments |
| `1_*.py` through `6_*.py` | Original vector construction, filtering, extraction, intervention, scoring, and layer scanning |
| `7_*.py` through `10_*.py` | Original layer/cosine, choice-response, and cross-dimensional analyses |
| `methology/persona/` | Five-dimensional definitions, prompt generator, and fixed extraction/evaluation materials |
| `runs/20261018_*/` | Five reference experiments: materials, vectors, retained pairs, and reduced per-answer scan records |
| `train/`, `comfort_validation/` | Train preprocessing, numerical inference helpers, and human-trade-off analysis |
| `swissmetro/` | Application preprocessing, probability readout, shared fitting, personal estimation, and their required helpers |
| `baselines/`, `pbel/`, `lora_sft/` | Prompt comparators, PBEL adaptation, and LoRA-SFT reference |
| `coefficient_calibration/` | Coefficient sensitivity and railway calibration experiments |
| `llama_validation/`, `qwen3b_validation/` | Cross-model construction and evaluation pipelines |
| `source_data/` | Compact numerical evidence, reference settings, and provenance |
| `tests/` | CPU tests of steering and shared scoring logic |

The numbered experiment filenames and reference run identifiers are retained so that saved settings and source references remain traceable. Repeated API diagnostics, raw API caches, exploratory runs, model checkpoints, virtual environments, and manuscript files are excluded.

## Install the model-experiment environment

The recorded research environment is Python 3.11.2 on Windows, PyTorch 2.7.1 with CUDA 12.8, and an NVIDIA RTX 5060 Ti with 16 GB VRAM. Exact Python package versions are in `requirements.txt` and `environment-tested.json`. The model experiments require an NVIDIA GPU; memory use depends on precision, context length, and batch size. The numerical quick start runs on CPU.

Install the appropriate PyTorch build for your system first. For the recorded CUDA build:

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
# Only for LoRA-SFT:
python -m pip install -r requirements-sft.txt
```

Download the main model separately:

```bash
hf download Qwen/Qwen2.5-7B-Instruct --local-dir models/Qwen2.5-7B-Instruct
```

The default model location is `models/Qwen2.5-7B-Instruct`. To use an existing local copy, set `TRAVEL_MODEL_PATH`. API-based contrast-pair filtering and response judging use DeepSeek; set `DEEPSEEK_API_KEY` in your shell.

```bash
# Linux/macOS
export TRAVEL_MODEL_PATH=/absolute/path/to/Qwen2.5-7B-Instruct
export DEEPSEEK_API_KEY=your_key
```

```powershell
# Windows PowerShell
$env:TRAVEL_MODEL_PATH = "D:/models/Qwen2.5-7B-Instruct"
$env:DEEPSEEK_API_KEY = "your_key"
```

`.env.example` lists the supported variables; it is a reference file and is not loaded automatically. Never place a real key in the source code. Full experiments can take hours and API judging incurs provider charges. Installation and runtime depend on download bandwidth and hardware; this package's local validation did not rerun the full GPU/API experiments.

## Construct and evaluate preference vectors

The fixed materials contain five high/low prompt pairs and 20 extraction scenarios per dimension, producing 100 response pairs. Each dimension has another 20 evaluation scenarios. The material language is Chinese, matching the experiments.

Run the stages in sequence for a fresh experiment:

```bash
python scripts/run_behavior.py --trait safety --stage generate
python scripts/run_behavior.py --trait safety --stage filter
python scripts/run_behavior.py --trait safety --stage extract
python scripts/run_behavior.py --trait safety --stage evaluate
python scripts/run_behavior.py --trait safety --stage judge
python scripts/run_behavior.py --trait safety --stage scan
```

Use any of `environment`, `safety`, `comfort`, `convenience`, or `flexibility`. New artifacts are saved under `runs/<timestamp>_<trait>/`; subsequent stages follow `runs/latest.txt`. Pass `--run-dir <run-name>` to select a particular run. `--batch-size` defaults to 32. Run one dimension at a time when using the latest-run pointer.

The public runner applies the paper's dimension-specific settings:

| Dimension | Physical layer | Extraction | Injection scope | Response re-encoding context |
| --- | ---: | --- | --- | --- |
| Comfort | 18 | `prompt_last` | `response` | Original profile |
| Environment | 16 | `prompt_last` | `prompt` | Original profile |
| Safety | 14 | `response_mean` | `all` | Original profile |
| Convenience | 14 | `response_mean` | `system` | Neutral |
| Flexibility | 18 | `response_mean` | `all` | Original profile |

Layers are numbered from one; the final block is measured before the model's final normalization. `response_mean` averages response-token states. `prompt_last` uses the final prompt position. The main scan uses physical layers 2, 4, ..., 28 and `alpha = [-1, 0, 1, 2, 3]`.

The filter requires high-concern scores of at least 50, low-concern scores below 50, factual consistency, and non-truncated responses. Preference scores and factual-quality flags are stored separately. Target-option choice rates must use the same matched question set across coefficients. Live API reruns can differ because of provider/model changes and stochastic judging; the released judgments support numerical reproduction of the reported results.

To rerun the original analysis and figure scripts on the included reference scans:

```bash
python 7_analyze_layers_and_cosine.py
python 8_plot_choice_effects.py
```

## Train and Swissmetro applications

Download and preprocess the public survey data:

```bash
python scripts/prepare_data.py --dataset all
```

This rebuilds the 2,929 Train records from 235 respondents and the eligible Swissmetro records with deterministic splits. Local source files can also be supplied with `--train-csv` and `--swissmetro-dat`. Sources, units, hashes, and license boundaries are documented in [docs/DATA.md](docs/DATA.md).

Regenerate the Train probabilities and repeat the original human-trade-off grouping/evaluation:

```bash
python scripts/collect_train.py
python comfort_validation/4_human_alignment.py
```

Train inference uses the existing float32-accumulation path with highest matmul precision and math attention. It evaluates a fixed comfort vector at `alpha = [-3, -1, 0, 1, 3]`. The grouping and evaluation records do not overlap.

Replay the complete Swissmetro prediction calculation from released raw state probabilities and frozen shared settings, including personal-weight estimation and the common history-frequency rule:

```bash
python scripts/run_swissmetro.py --cohort development --replay
python scripts/run_swissmetro.py --cohort confirmation --style synonyms --replay
python scripts/run_swissmetro.py --cohort confirmation --style punctuation --replay
```

This replay needs PyTorch and the prepared Swissmetro data, but no model weights or API. It verifies agreement with the archived final probabilities and checks that changing held-out labels cannot affect personal weights. To recompute the raw state probabilities with the local model, omit `--replay`. Prediction outputs go to `outputs/swissmetro/`.

The Swissmetro application uses seven saved injection configurations, shared attribute adjustment, and individual MAP mixture weights estimated from six known choices. When all six known choices agree, the saved common frequency rule supplies the prediction. `swissmetro/75_joint_state_attribute_fit.py` contains shared-parameter fitting; `51_personal_mixture_weights.py` contains personal estimation. [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) maps the remaining comparators and appendix experiments to their source code.

## Validation

```bash
python scripts/check_release.py
python -m unittest discover -s tests -v
```

The release check covers required files, Python syntax, material fingerprints, file checksums, and credential patterns. The tests cover intervention positions, vector-extraction conventions, choice parsing, and scoring/cache parity with mocked API responses. They do not send API requests or load a language model.

## Citation and license

Citation metadata is provided in [CITATION.cff](CITATION.cff). The manuscript is prepared for submission; no publication DOI or repository DOI is assigned here.

Original TRAVEL code is released under the [MIT License](LICENSE). Third-party data, pretrained models, and adapted upstream material retain their own terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The SynthesizeMe license is preserved in `licenses/`.

Relevant methodological sources include [Persona Vectors](https://github.com/safety-research/persona_vectors), [PERSONA](https://github.com/xcfcode/persona), [SynthesizeMe](https://github.com/SALT-NLP/SynthesizeMe), and the [PBEL paper](https://doi.org/10.1287/trsc.2025.0330). Travel-preference constructs are adapted from [Vredin Johansson et al. (2006)](https://doi.org/10.1016/j.tra.2005.09.001); the synthetic materials are this study's operationalization.
