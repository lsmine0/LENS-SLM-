# Ensemble Reasoning with SLMs & CMLP (Confidence Multi Layer Perceptron)
## Środowisko

__Główne urządzenie__: AMD Radeon RX 7900 XTX (trening, benchmarking oraz inferencja).

__Zarządzanie pakietami__: środowisko zarządzane przez `Conda`, z zależnościami zdefiniowanymi w pliku `environment.yml`.

__Frameworki__: projekt oparty na bibliotekach __Hugging Face__ `transformers` oraz `datasets`, zapewniających spójne ładowanie modeli oraz tokenizację.

## Dane treningowe

Confidence Multi-Layer Perceptron (CMLP) jest trenowany na wyselekcjonowanej mieszance zbiorów danych opartych na zadaniach logicznych. Główny nacisk położono na zadania binarne oraz spójność logiczną, zamiast złożonych formatów wielokrotnego wyboru, aby utrzymać czysty i stabilny sygnał nadzorowania.

| Zbiór danych | Podzbiór | Typ rozumowania |
|---|---|---|
| `google/boolq` | `train[:2000]` | rozumowanie binarne (tak/nie) |
| `smoorsmith/prontoqa` | `train[:500]` | wieloetapowa logika |
| `skrishna/coin_flip` | `train[:2000]` | spójność probabilistyczna |
| `tasksource/proofwriter` | `train[:2000]` | strukturalne dowodzenie |

__Uwaga__: zbiory takie jak __SWAG__ oraz __MathQA__ zostały pominięte, ponieważ ich złożone formaty odpowiedzi nie były zgodne z uproszczonym podejściem do etykietowania opartego na pewności.

## Modele

Ensemble został zbudowany w oparciu o kilka modeli typu __Llama 3.2 SLM (Small Language Models)__. Wybór modeli z tej samej rodziny tokenizerów umożliwia precyzyjne __ensemblowanie na poziomie tokenów__.

Uwzględnione modele:

- `unsloth/Llama-3.2-1B-Instruct`  
- `ai-nexuz/llama-3.2-1b-instruct-fine-tuned`  
- `NousResearch/Hermes-3-Llama-3.2-3B`  
- `keeeeenw/Llama-3.2-1B-Instruct-Open-R1-Distill`  
- `EpistemeAI/Reasoning-Llama-3.2-1B-Instruct-v1.2`  

## Metodologia benchmarku

Benchmark porównuje wydajność pojedynczych modeli SLM z wynikiem całego ensemble na stałej próbce danych (np. `train[:200]`).

__Pipeline inferencji__

1. __Generowanie:__ każdy model generuje predykcję na podstawie końcowych logitów.  
2. __Ekstrakcja__: odpowiedzi binarne (True/False) są mapowane za pomocą zdefiniowanych tokenów.  
3. __Scoring pewności__: __CMLP__ generuje ocenę pewności na podstawie stanów ukrytych i pewności wyjścia modelu.  
4. __Ważenie__: wyniki są normalizowane funkcją Softmax w celu uzyskania wag ensemble.  
5. __Agregacja__: predykcje są łączone jako ważona suma prawdopodobieństw.

__Funkcja decyzji__

Predykcja ensemble definiowana jest jako:

$$
P_{ensemble} = \sum_{i=1}^{n} (w_i \cdot P_i)
$$

gdzie $w_i$ to waga wyznaczona przez CMLP, a $P_i$ to prawdopodobieństwo zwrócone przez dany model.

## Wyniki

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