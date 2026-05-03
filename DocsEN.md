# Ensemble Reasoning with SLMs & CMLP (Confidence Multi Layer Perceptron)
## Environment

__Primary Hardware__: AMD Radeon RX 7900 XTX (Training, Benchmarking, and Inference).

__Package Management__: Managed via `Conda` with dependencies defined in `environment.yml`.

__Frameworks__: Built on __Hugging Face__ `transformers` and `datasets` for consistent model loading and tokenization.

## Training Data

The Confidence Multi-Layer Perceptron (CMLP) is trained on a curated mixture of logic-based datasets. The focus is on binary and logical consistency tasks rather than complex multiple-choice formats in order to maintain a clean and reliable supervision signal.

| Dataset | Subset | Reasoning Type |
|---|---|---|
| `google/boolq` | `train[:2000]` | Boolean (Yes/No) Reasoning |
| `smoorsmith/prontoqa` | `train[:500]` | Multi-step Logic |
| `skrishna/coin_flip` | `train[:2000] `| Probabilistic Consistency |
| `tasksource/proofwriter` | `train[:2000]` | Structured Proof Generation |

__Note__: Datasets like __SWAG__ and __MathQA__ were excluded because their complex answer formats did not align with our simplified confidence-based labeling approach.

## Models

The ensemble is built using multiple __Llama 3.2–based small language models (SLMs)__, Selecting models from the same tokenizer family allows for precise __token-level ensembling__.

Included Models:

- `unsloth/Llama-3.2-1B-Instruct`  
- `ai-nexuz/llama-3.2-1b-instruct-fine-tuned`  
- `NousResearch/Hermes-3-Llama-3.2-3B`  
- `keeeeenw/Llama-3.2-1B-Instruct-Open-R1-Distill`  
- `EpistemeAI/Reasoning-Llama-3.2-1B-Instruct-v1.2`  

## Benchmark Methodology

The benchmark compares individual SLM performance against the combined ensemble output across a fixed subset of samples (e.g., train[:200]).

__The Inference Pipeline__
1. __Generation__: Each model generates predictions using its final token logits.

2. __Extraction__: Binary decisions (True/False) are mapped via predefined token IDs.

3. __Confidence Scoring__: The __CMLP__ generates a score for each model based on its hidden states and output certainty.

4. __Weighting__: Scores are normalized via Softmax to create ensemble weights.

5. __Aggregation__: Predictions are combined using a weighted probability sum.

__Decision Formula__

The ensemble prediction is defined as the highest weighted probability across all participating models:

$$
P_{ensemble} = \sum_{i=1}^{n} (w_i \cdot P_i)
$$

Where $w_i$ is the CMLP-derived weight and $P_i$ is the individual model probability.

## Results
```bash
--- BENCHMARK COMPARISON (Accuracy %) ---
Dataset                                boolq  coin_flip  prontoqa  proofwriter
Model
Ensemble                                60.5       47.0      42.5         83.0
Hermes-3-Llama-3.2-3B                   53.0       47.0      42.5         53.0
Llama-3.2-1B-Instruct                   61.5       47.0      60.5         68.0
Llama-3.2-1B-Instruct-Open-R1-Distill   55.5       55.5      27.0         92.5
Reasoning-Llama-3.2-1B-Instruct-v1.2    60.0       47.0      12.5         70.5
llama-3.2-1b-instruct-fine-tuned        61.5       47.0      50.0         61.5
```
