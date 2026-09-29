"""Publish a local ICA Lens artifact to a Hugging Face Model repository."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from dotenv import dotenv_values
from huggingface_hub import HfApi

from icalens import ICALens

DEFAULT_ENV_FILE = Path(".env")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="icalens publish", description=__doc__)
    parser.add_argument(
        "repo_id", help="Destination Model repository, for example sida/icalens-gpt2-small."
    )
    parser.add_argument("--lens", type=Path, required=True, help="Local ICA Lens directory.")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help="File containing HF_TOKEN (default: .env in the current directory).",
    )
    parser.add_argument("--private", action="store_true", help="Create a private repository.")
    parser.add_argument("--revision", default="main", help="Destination branch (default: main).")
    parser.add_argument(
        "--commit-message",
        default="Upload ICA Lens artifacts",
        help="Hugging Face commit message.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    lens_path = args.lens.expanduser().resolve()
    lens = ICALens.from_pretrained(lens_path)
    token = load_hf_token(args.env_file)

    api = HfApi(token=token)
    try:
        account = api.whoami()
    except Exception as error:
        raise RuntimeError(
            "Hugging Face authentication failed. Run `hf auth login` or set HF_TOKEN."
        ) from error
    username = account.get("name", "unknown")
    print(f"Authenticated with Hugging Face as {username}.")
    print(f"Publishing {lens_path} to {args.repo_id}@{args.revision}...")

    result = lens.push_to_hub(
        args.repo_id,
        private=True if args.private else None,
        token=token,
        revision=args.revision,
        commit_message=args.commit_message,
    )
    cloud = ICALens.from_pretrained(
        args.repo_id,
        revision=args.revision,
        token=token,
        force_download=True,
    )
    if cloud.metadata != lens.metadata:
        raise RuntimeError("Uploaded manifest does not match the local artifact manifest.")
    print(f"Verified uploaded manifest and available layers {cloud.available_layers}.")
    print(result)


def load_hf_token(env_file: Path) -> str | None:
    """Read HF_TOKEN from a dotenv file, falling back to standard Hub auth."""
    path = env_file.expanduser().resolve()
    if not path.is_file():
        return None
    token = dotenv_values(path).get("HF_TOKEN")
    if token is None:
        return None
    token = token.strip()
    if not token:
        raise ValueError(f"HF_TOKEN is empty in {path}")
    return token


if __name__ == "__main__":
    main()
