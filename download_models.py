"""Download the three permitted models into models/. Run once; needs internet."""
from huggingface_hub import snapshot_download

for repo, out in [("openai/whisper-small",        "models/whisper-small"),
                  ("Qwen/Qwen2.5-0.5B",           "models/qwen2.5-0.5b"),
                  ("facebook/wav2vec2-base-960h", "models/wav2vec2-base-960h")]:
    snapshot_download(repo, local_dir=out)
    print("ok:", out)
