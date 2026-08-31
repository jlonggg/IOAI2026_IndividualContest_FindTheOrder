"""Find The Order — full pipeline. Target: < 10 min on 1 GPU."""
import os, json, glob, time
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
import numpy as np, torch, librosa
from transformers import (WhisperForConditionalGeneration, WhisperProcessor,
                          AutoModelForCausalLM, AutoTokenizer)

ROOT     = "."
TEST_DIR = f"{ROOT}/dataset/test_private"
OUTPUT   = "submission.json"
WHISPER  = f"{ROOT}/models/whisper-small"
QWEN     = f"{ROOT}/models/qwen2.5-0.5b"
HEAD     = "The following is a conversation between two people.\n\n"
BEAM, TAU, SK_ITERS, ASR_BS, TOPK = 1000, 0.25, 20, 32, 8
dev = "cuda" if torch.cuda.is_available() else "cpu"
T0 = time.time()
def log(m): print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)

# ---------- 1. inventory ----------
dids = sorted((d for d in os.listdir(TEST_DIR) if os.path.isdir(f"{TEST_DIR}/{d}")),
              key=lambda x: int(x) if x.isdigit() else x)
nch = {d: len(glob.glob(f"{TEST_DIR}/{d}/chunk_*.wav")) for d in dids}
prefix = json.load(open(f"{TEST_DIR}/prefix.json"))
files = [(d, c) for d in dids for c in range(nch[d])]
log(f"{len(dids)} dialogues, {len(files)} chunks")

# ---------- 2. one audio pass: Whisper transcript + MFCC speaker feature ----------
wproc = WhisperProcessor.from_pretrained(WHISPER)
wmod  = WhisperForConditionalGeneration.from_pretrained(
            WHISPER, torch_dtype=torch.float16).to(dev).eval()
text, mfcc = {}, {}
for i in range(0, len(files), ASR_BS):
    b = files[i:i+ASR_BS]
    auds = [librosa.load(f"{TEST_DIR}/{d}/chunk_{c}.wav", sr=16000)[0] for d, c in b]
    for (d, c), y in zip(b, auds):
        m = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=20)
        mfcc[(d, c)] = np.concatenate([m.mean(1), m.std(1)])
    feats = wproc(auds, sampling_rate=16000, return_tensors="pt").input_features
    with torch.no_grad():
        ids = wmod.generate(feats.half().to(dev), language="en", task="transcribe",
                            max_new_tokens=100)
    for (d, c), t in zip(b, wproc.batch_decode(ids, skip_special_tokens=True)):
        text[(d, c)] = t.strip()
    if i % (ASR_BS*10) == 0: log(f"asr {i+len(b)}/{len(files)}")
del wmod; torch.cuda.empty_cache()
log("asr done")

# ---------- 3. speaker split: dialogues strictly alternate A-B-A-B ----------
def split_spec(X, p0, p1):
    """Group taking the EVEN positions (starts at p0). Spectral, size-constrained."""
    n = len(X); S = X @ X.T
    e = np.linalg.eigh(S - np.diag(np.diag(S)))[1][:, -1]
    if e[p0] < e[p1]: e = -e
    o = sorted(range(n), key=lambda c: -e[c]); o.remove(p0); o.remove(p1)
    return set([p0] + o[:(n+1)//2 - 1])

groups = {}
for d in dids:
    n = nch[d]
    X = np.array([mfcc[(d, c)] for c in range(n)])
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    p0, p1 = prefix[d]
    A = split_spec(X, p0, p1)
    groups[d] = (A, set(range(n)) - A, p0, p1, n)
log("speaker split done")

# ---------- 4. Qwen pairwise adjacency score ----------
tok = AutoTokenizer.from_pretrained(QWEN)
if tok.pad_token is None: tok.pad_token = tok.eos_token
qwen = AutoModelForCausalLM.from_pretrained(QWEN, torch_dtype=torch.float16).to(dev).eval()

@torch.no_grad()
def logp(ctxs, conts):
    """Mean per-token logprob of each continuation given its context."""
    ci = [tok(c).input_ids for c in ctxs]; ki = [tok(k).input_ids for k in conts]
    ids = [a + b for a, b in zip(ci, ki)]; L = max(map(len, ids))
    inp = torch.full((len(ids), L), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((len(ids), L), dtype=torch.long)
    lab = torch.full((len(ids), L), -100, dtype=torch.long)
    for j, x in enumerate(ids):
        inp[j, :len(x)] = torch.tensor(x); att[j, :len(x)] = 1
        lab[j, len(ci[j]):len(x)] = torch.tensor(x[len(ci[j]):])
    inp, att, lab = inp.to(dev), att.to(dev), lab.to(dev)
    lg = qwen(input_ids=inp, attention_mask=att).logits
    tot = torch.zeros(len(ids), device=dev)
    CH = max(1, int(4e7 // (lg.size(1) * lg.size(2))))   # cap the fp32 upcast buffer
    for a in range(0, len(ids), CH):
        z = lg[a:a+CH]
        nll = torch.nn.functional.cross_entropy(
            z[:, :-1].reshape(-1, z.size(-1)).float(), lab[a:a+CH, 1:].reshape(-1),
            reduction="none", ignore_index=-100).view(z.size(0), -1)
        tot[a:a+CH] = -nll.sum(1)
    del lg
    return [(tot[j] / len(ki[j])).item() for j in range(len(ids))]

def batched(ctxs, conts, tok_budget=8000, max_bs=64):
    """Batch by token budget, not by count -- a fixed count OOMs on long turns."""
    n = len(ctxs); ln = [len(ctxs[i]) + len(conts[i]) for i in range(n)]
    order = sorted(range(n), key=lambda i: ln[i]); res = [0.0] * n; i = 0
    while i < n:
        j = i; mx = 0
        while j < n and j - i < max_bs:
            m = max(mx, ln[order[j]] // 3 + 8)          # ~3 chars per token
            if (j - i + 1) * m > tok_budget and j > i: break
            mx = m; j += 1
        idx = order[i:j]
        for k, v in zip(idx, logp([ctxs[k] for k in idx], [conts[k] for k in idx])):
            res[k] = v
        i = j
    return res

def turn(spk, t): return f"{spk}: {t}\n"

def score_matrix(d):
    A, B, p0, p1, n = groups[d]
    tx = {c: text[(d, c)] for c in range(n)}
    lbl = {c: ("A" if c in A else "B") for c in range(n)}
    pairs = [(u, v) for u in A for v in B] + [(v, u) for u in A for v in B]
    vals = batched([HEAD + turn(lbl[u], tx[u]) for u, v in pairs],
                   [turn(lbl[v], tx[v]) for u, v in pairs])
    S = np.full((n, n), -1e9)
    for (u, v), s in zip(pairs, vals): S[u][v] = s
    return S

def sinkhorn(S, mask, iters=SK_ITERS, tau=TAU):
    """Make the score matrix doubly stochastic: each turn has exactly one successor
    and one predecessor, so per-utterance popularity biases must be normalised out.
    This replaces the earlier PMI correction, which did the same job crudely."""
    L = np.where(mask, S / tau, -1e9); L = L - L.max()
    for _ in range(iters):
        L = L - np.where(mask, np.log(np.exp(np.where(mask, L, -1e9)).sum(1, keepdims=True) + 1e-30), 0)
        L = np.where(mask, L, -1e9)
        L = L - np.where(mask, np.log(np.exp(np.where(mask, L, -1e9)).sum(0, keepdims=True) + 1e-30), 0)
        L = np.where(mask, L, -1e9)
    return L

def beam_order(S, d, topk=1):
    """Vectorised beam search, forced to alternate speakers, anchored on prefix."""
    A, B, p0, p1, n = groups[d]
    gA = np.array(sorted(A)); gB = np.array(sorted(B))
    sc = np.array([S[p0][p1]]); last = np.array([p1])
    mask = np.array([(1 << p0) | (1 << p1)], dtype=np.int64)
    seq = np.array([[p0, p1]])
    for pos in range(2, n):
        cand = gA if pos % 2 == 0 else gB
        free = ((mask[:, None] >> cand[None, :]) & 1) == 0
        ns = np.where(free, sc[:, None] + S[last[:, None], cand[None, :]], -np.inf).ravel()
        k = min(BEAM, int(free.sum()))
        top = np.argpartition(-ns, k - 1)[:k] if k < ns.size else np.arange(ns.size)
        top = top[np.argsort(-ns[top])]
        bi, ci = top // len(cand), top % len(cand)
        sc = ns[top]; nc = cand[ci]
        seq = np.concatenate([seq[bi], nc[:, None]], 1)
        mask = mask[bi] | (np.int64(1) << nc.astype(np.int64)); last = nc
    return [[int(x) for x in seq[i]] for i in range(min(topk, len(seq)))]

# ---------- 5. full-context rerank ----------
def rerank(cands, d):
    """The adjacency objective only ever sees one turn back, so its top-1 is often
    beaten by another candidate in its own beam (oracle over top-16 is ~0.95 vs
    ~0.81 for top-1). Rescore each candidate ordering under the WHOLE dialogue."""
    A, B, p0, p1, n = groups[d]
    tx = {c: text[(d, c)] for c in range(n)}
    lbl = {c: ("A" if c in A else "B") for c in range(n)}
    ctxs, conts, owner = [], [], []
    for oi, o in enumerate(cands):
        for i in range(1, len(o)):
            ctxs.append(HEAD + "".join(turn(lbl[c], tx[c]) for c in o[:i]))
            conts.append(turn(lbl[o[i]], tx[o[i]]))
            owner.append(oi)
    tot = np.zeros(len(cands))
    for oi, v in zip(owner, batched(ctxs, conts, tok_budget=3000)):
        tot[oi] += v
    return cands[int(tot.argmax())]

# ---------- 6. predict ----------
answers = {}
for k, d in enumerate(dids):
    S = score_matrix(d)
    cands = beam_order(sinkhorn(S, S > -1e8), d, topk=TOPK)
    order = rerank(cands, d) if len(cands) > 1 else cands[0]
    P = [0] * nch[d]
    for pos, c in enumerate(order): P[c] = pos
    assert sorted(P) == list(range(nch[d])), f"invalid permutation for {d}"
    answers[d] = P
    if k % 20 == 0: log(f"order {k+1}/{len(dids)}")

json.dump(answers, open(OUTPUT, "w"))
log(f"wrote {OUTPUT}: {len(answers)} dialogues")
