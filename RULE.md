# 🔀 Find The Order

> English translation of the official Vietnamese problem statement, provided for readers of this
> repository. In case of any discrepancy, the organizers' original text is authoritative.

This task uses audio and relatively large pretrained models. If your personal machine lacks a
suitable GPU, you are **encouraged to use Google Colab or Kaggle** to develop and test your solution.

📦 **Dataset & Baseline:** Download Dataset & Baseline

---

## 1. Background & Goal

In spoken language processing systems, a conversation is made up of many turns spoken by different
participants. The order of those turns determines the meaning and coherence of the whole conversation.

In this task you are given English conversations between two participants, **Speaker A** and
**Speaker B**. Each turn is stored as a separate `.wav` audio file.

However, the turns have been **randomly shuffled**. The filename `chunk_{k}.wav` only denotes the
index of the chunk within the shuffled dataset — it says nothing about the chunk's true position in
the conversation.

**Your task is to recover the original temporal order of the entire conversation.**

---

## 2. Dataset

Each dialogue consists of `n` audio files:

```
chunk_0.wav
chunk_1.wav
...
chunk_{n-1}.wav
```

Each chunk contains exactly one turn from one of the two speakers. The chunks are shuffled, and
dialogues range from **7 to 20 turns**.

The audio files have the following properties:
- Mono
- 44.1 kHz sample rate
- May be resampled if needed

### Starting-point information

The file `prefix.json` provides **the first two chunks in their correct conversational order**.

For example:

```json
{
  "11": [7, 12]
}
```

means that for dialogue `11`:

```
chunk_7.wav → chunk_12.wav → ...
```

are the first two turns of the conversation.

This information fixes the starting point and removes the ambiguity between reading the conversation
forwards or backwards.

---

## 3. Dataset Structure

The dataset is split into several sets serving different purposes:

| Directory | `answers.json` | Purpose |
|---|---|---|
| `dataset/train/` | ✅ | Training / fine-tuning |
| `dataset/pretrain/` | ✅ | Additional pretraining / training |
| `dataset/pretest/` | ✅ | Validation / local testing |
| `dataset/test_private/` | ❌ | Final inference and submission |

### Directory layout

```
dataset/
├── train/
│   ├── answers.json
│   ├── prefix.json
│   └── <dialogue_id>/
│       ├── chunk_0.wav
│       ├── ...
│       └── chunk_{n-1}.wav
│
├── pretrain/
│   ├── answers.json
│   ├── prefix.json
│   └── <dialogue_id>/
│       ├── chunk_0.wav
│       ├── ...
│       └── chunk_{n-1}.wav
│
├── pretest/
│   ├── answers.json
│   ├── prefix.json
│   └── <dialogue_id>/
│       ├── chunk_0.wav
│       ├── ...
│       └── chunk_{n-1}.wav
│
└── test_private/
    ├── prefix.json
    └── <dialogue_id>/
        ├── chunk_0.wav
        ├── ...
        └── chunk_{n-1}.wav
```

Where:
- `prefix.json` holds the first two chunks of each dialogue.
- `answers.json` holds the correct order of the chunks.
- `answers.json` is **not provided** for `test_private`.

---

## 4. The Task

For each dialogue, determine the **temporal position** of every chunk.

The result for a dialogue must be a permutation `P` of:

```
{0, 1, ..., n-1}
```

where:

```
P[i]
```

is the temporal position of `chunk_i.wav`.

Convention:
- `0`: first chunk
- `1`: second chunk
- `...`
- `n-1`: last chunk

### Example

A dialogue with 3 chunks:

| Chunk | Content | Correct position |
|---|---|---|
| `chunk_0.wav` | "No worries — I'll send you the notes afterwards." | 2 |
| `chunk_1.wav` | "Hey, are you coming to the three o'clock meeting?" | 0 |
| `chunk_2.wav` | "I can't — I've got a dentist appointment then." | 1 |

The correct order is:

```
chunk_1 → chunk_2 → chunk_0
```

Therefore:

```
[2, 0, 1]
```

is the answer for this dialogue.

The corresponding `prefix.json` would contain:

```
[1, 2]
```

---

## 5. Output

You must produce a JSON file containing predictions for every dialogue in:

```
dataset/test_private/
```

For example:

```json
{
  "17": [2, 0, 1],
  "42": [0, 1],
  "108": [3, 1, 0, 4, 2, 5]
}
```

Where:
- The key is the `dialogue_id`.
- The value is the predicted permutation for that dialogue.

---

## 6. Submission Format

You must submit **a single `.zip` file**.

For example:

```
submission.zip
```

The ZIP must contain both the source code and the predictions:

```
submission.zip
├── solution.ipynb
└── submission.json
```

- **`solution.ipynb`**: your source code, which must contain all the logic needed to produce the
  predictions. The organizers will check that `solution.ipynb` exists in the submission. The notebook
  should be able to run standalone in the contest environment, using the provided dataset and the
  permitted models.

- **`submission.json`**: the predictions for every dialogue in `dataset/test_private/`. For example:

```json
{
  "17": [2, 0, 1],
  "42": [0, 1],
  "108": [3, 1, 0, 4, 2, 5]
}
```

The JSON file must be named `submission.json`.

### Required structure

A valid submission must contain at least:

```
submission.zip
├── solution.ipynb
└── submission.json
```

> ⚠️ **Do not submit the notebook or the JSON on its own.** Package both files into `submission.zip`.
>
> ⚠️ `solution.ipynb` is required so that the submission includes source code and can be
> inspected/reproduced.

---

## 7. Prediction Rules

`P` must be a **valid permutation**:
- Its length must equal `n`.
- Every value appears exactly once.
- Values lie in `[0, n-1]`.
- Use **0-based** indexing.
- No dialogue may be omitted.

If a permutation is invalid, that dialogue scores **0**.

A dialogue missing from `submission.json`, an invalid JSON file, or malformed output data likewise
scores **0** for the dialogue in question.

---

## 8. Scoring

The task is scored by **pairwise ordering accuracy**.

For every pair of chunks, the system checks whether the two chunks are placed in the correct temporal
order.

A dialogue with `n` chunks has:

$$M = \frac{n(n-1)}{2}$$

chunk pairs.

Let `I` be the number of pairs inverted relative to the ground truth. The dialogue's score is:

$$score = 1 - \frac{I}{M}$$

Hence:

```
0 ≤ score ≤ 1
```

### Final score

The final score is the mean score across all dialogues:

$$\frac{1}{N}\sum_{d=1}^{N} score_d$$

where `N` is the number of dialogues being scored.

---

## 9. Permitted Models

You may **only use the following pretrained models**, during both training and inference:

### Speech representation
- **wav2vec 2.0**: you may use wav2vec 2.0 embeddings to represent audio content.

### Automatic Speech Recognition
- **OpenAI Whisper**, at any model size provided in the contest environment. Whisper may be used to
  convert speech to text and to exploit information in the conversation's content.

### Language model
- **Qwen2.5-0.5B**. It may be used:
  - zero-shot; or
  - fine-tuned on the provided datasets.

The permitted models are already downloaded in the environment.

> ⚠️ **No pretrained model outside the list above may be used.**
>
> ⚠️ **The 10-minute total budget** includes both training/fine-tuning and inference.

---

## 10. Environment Limits

The entire program must run within:
- **Time:** 10 minutes maximum.
- **GPU:** 1 GPU, roughly 16 GB VRAM.
- **Internet:** none.
- **Storage:** 5 GB.
- **Submission size:** no larger than 1 MB.

The time limit covers the whole process required to produce the final result, including:
1. Model initialization.
2. Data loading.
3. Training/fine-tuning, if any.
4. Feature extraction.
5. Inference.
6. Writing `submission.json`.

---

## 11. Local Testing

The `train`, `pretrain` and `pretest` sets all come with `answers.json`, so you can use them for
development and testing.

One possible workflow:

```
dataset/train/
dataset/pretrain/
        ↓
    training / fine-tuning
        ↓
dataset/pretest/
        ↓
    local evaluation
        ↓
dataset/test_private/
        ↓
    final inference
        ↓
submission.json
        ↓
submission.zip
```

The `test_private` set has *no* `answers.json` and is used for the final inference.

---

## 12. How to Submit

1. Develop your solution using `dataset/train/`, `dataset/pretrain/` and `dataset/pretest/`.
2. Test your solution and make sure `solution.ipynb` runs correctly.
3. Run inference on `dataset/test_private/`.
4. Produce:

```
submission.json
```

5. Put the source code and the predictions in the same directory:

```
submission/
├── solution.ipynb
└── submission.json
```

6. Compress them into:

```
submission.zip
```

7. Verify the submission:

```
submission.zip
├── solution.ipynb  ← required
└── submission.json ← required
```

8. Make sure the ZIP file does not exceed **1 MB**.
9. Upload `submission.zip` to the grading system.

> **Important:** the submission is not just predictions. Both `solution.ipynb` and `submission.json`
> are required inside the ZIP file.
