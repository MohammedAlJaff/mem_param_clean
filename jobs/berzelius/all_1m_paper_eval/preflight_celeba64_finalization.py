"""Check one frozen evaluation command without changing its outputs."""

import argparse
import json
import shlex
from pathlib import Path

from mem_param_src.runners import image_eval, image_protocol
from mem_param_src.runners.artifacts import read_json


_IGNORED_MANIFEST_FIELDS = {
    "created_at",
    "elapsed_seconds",
    "failure",
    "status",
    "updated_at",
}


class _IdentityMatched(Exception):
    pass


def check_command(command_file):
    """Build the requested identity and compare it with the saved manifest."""
    command_file = Path(command_file).expanduser().resolve()
    tokens = shlex.split(command_file.read_text(encoding="utf-8"))
    expected_prefix = [
        "python",
        "-u",
        "-B",
        "-m",
        "mem_param_src.runners.image_eval",
    ]
    if tokens[: len(expected_prefix)] != expected_prefix:
        raise ValueError("The frozen evaluator command has an unexpected prefix.")

    def compare(root, identity):
        manifest_path = Path(root) / "eval_manifest.json"
        existing = read_json(manifest_path)
        comparable = {
            key: value
            for key, value in existing.items()
            if key not in _IGNORED_MANIFEST_FIELDS
        }
        if comparable != identity:
            keys = sorted(set(comparable).union(identity))
            differences = {
                key: {
                    "saved": comparable.get(key, "<missing>"),
                    "requested": identity.get(key, "<missing>"),
                }
                for key in keys
                if comparable.get(key, "<missing>")
                != identity.get(key, "<missing>")
            }
            raise ValueError(
                "Frozen request identity differs from the saved manifest:\n"
                + json.dumps(differences, indent=2, sort_keys=True)
            )
        print(f"identity_match\t{manifest_path}")
        raise _IdentityMatched

    original = image_protocol._prepare_manifest
    image_protocol._prepare_manifest = compare
    try:
        image_eval.main(tokens[len(expected_prefix) :])
    except _IdentityMatched:
        return 0
    finally:
        image_protocol._prepare_manifest = original
    raise RuntimeError("The evaluator passed the manifest check unexpectedly.")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command-file", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return check_command(args.command_file)


if __name__ == "__main__":
    main()
