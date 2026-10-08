"""Tag the current Hub version, then upload a new adapter over it.

Run `huggingface_hub.login()` (or set HF_TOKEN) first.

    python push_to_hub.py --tag-current v1-3k-1epoch                 # only tag what is on the Hub now
    python push_to_hub.py --adapter runs/v2/adapter --tag v2-full    # upload v2, then tag it

Older versions stay reachable by tag, e.g.
    PeftModel.from_pretrained(base, REPO, revision="v1-3k-1epoch")
"""

import argparse

from huggingface_hub import HfApi

p = argparse.ArgumentParser()
p.add_argument("--repo", default="AnqiXaq/dialogsum-qlora-adapter")
p.add_argument("--tag-current", default=None, help="tag the Hub's current main before anything else")
p.add_argument("--adapter", default=None, help="local adapter folder to upload")
p.add_argument("--tag", default=None, help="tag to put on the uploaded version")
p.add_argument("--message", default=None)
args = p.parse_args()

api = HfApi()
if args.tag_current:
    api.create_tag(args.repo, tag=args.tag_current, exist_ok=True)
    print(f"Tagged current main as {args.tag_current}")

if args.adapter:
    # The README (model card) is maintained on the Hub, so do not overwrite it.
    api.upload_folder(
        repo_id=args.repo,
        folder_path=args.adapter,
        ignore_patterns=["README.md", "checkpoint-*"],
        commit_message=args.message or f"Upload adapter from {args.adapter}",
    )
    print(f"Uploaded {args.adapter} to {args.repo}")
    if args.tag:
        api.create_tag(args.repo, tag=args.tag, exist_ok=True)
        print(f"Tagged as {args.tag}")
