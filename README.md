# Find The Order

Recover the temporal order of turns in a two-person conversation (OLP AI 2026). Each turn is a
separate `.wav` file, shuffled; the output is a permutation where `P[i]` is the temporal position of
`chunk_i.wav`.

**Official score: 84.4/100** (internal validation over 290 dialogues: 0.8592).

📊 **[Full technical write-up](https://claude.ai/code/artifact/ce2abbc1-d0f2-4631-9d7c-b0a5dd47c608)** —
data analysis, figures, and all 14 directions that were tried and failed.

---

## Results

| | val 290 | official |
|---|---|---|
| identity permutation (baseline) | ~0.50 | — |
| speaker split + alternation only | 0.709 | — |
| + adjacency PMI, beam 200 | 0.7950 | 0.78445 |
| + Sinkhorn τ=0.25, beam 1000 | 0.8111 | — |
| **+ full-context rerank top-8** | **0.8592** | **0.844** |
| oracle top-8 (ceiling of the pool) | 0.9294 | — |

End-to-end on `test_private` (100 dialogues, 1079 chunks, 1× RTX 2080 Ti): **132 s** against a
600 s budget.

## Method

Four layers, no model fine-tuning:

1. **Speaker alternation.** Conversations alternate strictly `A-B-A-B` — MFCC cosine similarity along
   the gold order is −0.38 at distance 1 and +0.28 at distance 2. Split the two speakers by spectral
   clustering over MFCC features, orient and anchor the split with `prefix.json`, and force the group
   sizes. Speaker assignment accuracy: **99.9%**. This layer alone moves the score 0.50 → 0.709.
2. **Content.** Whisper-small transcribes each chunk, sharing a single audio pass with the MFCC
   extraction above.
3. **Ordering.** Qwen2.5-0.5B scores `S[u][v] = mean logP(v|u)`; **Sinkhorn** normalization makes the
   matrix doubly stochastic (each turn has exactly one successor); beam search of width 1000 forces
   the alternation.
4. **Rerank.** The adjacency objective only ever looks one turn back, so the beam's top-1 is often
   beaten by another candidate from its own beam. Rescore the top-8 candidates under the **whole
   conversation** as context and take the best.

## Files

| File | Contents |
|---|---|
| `baseline.ipynb` | Starting baseline: identity permutation + demos of the three permitted models |
| `solution_final.ipynb` | **Final solution** — the submitted version, split by layer |
| `pipeline.py` | Same code as `solution_final.ipynb`, as a single-command script |
| `submission.json` | The output that scored 84.4 |
| `RULE.md` | The original problem statement (English translation) |

## Setup

Tested on Python 3.8, CUDA 12.1, 1× RTX 2080 Ti (11 GB).

```bash
pip install -r requirements.txt
```

### Models (3.5 GB — not in the repository)

The contest permits only the three checkpoints below. Download them into `models/`:

```bash
pip install huggingface_hub
python download_models.py
```

Or manually:

| Directory | HuggingFace repo |
|---|---|
| `models/whisper-small` | `openai/whisper-small` |
| `models/qwen2.5-0.5b` | `Qwen/Qwen2.5-0.5B` |
| `models/wav2vec2-base-960h` | `facebook/wav2vec2-base-960h` |

The pipeline runs with `HF_HUB_OFFLINE=1`, so the download must complete before you run it.

### Dataset (4.2 GB of audio — not in the repository)

The repository carries the labels only: `dataset/*/answers.json` and `dataset/*/prefix.json`. Download
the `.wav` audio from the contest site and unpack it into this layout:

```
dataset/
├── train/        answers.json, prefix.json, <dialogue_id>/chunk_{k}.wav   (1288 dialogues)
├── pretrain/     answers.json, prefix.json, ...                           (20 dialogues)
├── pretest/      answers.json, prefix.json, ...                           (20 dialogues)
└── test_private/ prefix.json, ...                                         (100 dialogues)
```

Audio is mono at 44.1 kHz (the pipeline resamples to 16 kHz). Each dialogue has 7–20 turns.
`prefix.json` gives the first two chunks **in their correct order** — this is the anchor, and it is
what removes the ambiguity between reading the conversation forwards and backwards.

## Running

```bash
python pipeline.py      # reads dataset/test_private/ → writes submission.json
```

Or open `solution_final.ipynb` and run the cells in order.

## Technical notes

Things that were **tried and lost**, recorded so they are not retried:

- **Fine-tuning Qwen** (both LM loss and listwise ranking loss): 0.8151 / 0.8046, both below 0.8589.
- **Backward LM**, or blending `fwd + w·bwd`: every `w > 0` drags the score down; a 3-dimensional grid
  search returns `(1, 0, 0)`.
- **Hill-climbing swaps within a speaker group**: the move set has an oracle of 0.9999, yet climbing
  makes the score fall.
- **PMI / positional normalization at the rerank layer**: cannot change a single digit. All candidates
  contain the same set of turns in a different order, so any per-utterance correction cancels out.
- **Iterated reranking** (feeding 2-turn context back into `S`): amplifies errors, oracle drops
  0.9294 → 0.8307.
- **Filtered reranking** (only the dialogues where the beam is least certain): 25%/50%/75% all score
  below reranking everything.

The remaining bottleneck is the **reranker**, not the ASR and not the beam: the correct candidate is
already sitting in the top-8 pool (oracle 0.9294), but the reranker picks it only 145/290 times, and
the best candidate is spread **evenly across ranks 1–7**. That shape rules out a calibration problem —
the scorer is simply blind on half the dialogues.
