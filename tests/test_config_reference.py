"""Keep the human-readable config reference synchronized with the base config."""

import unittest
from pathlib import Path

from mem_param_src.config import config_to_dict, get_base_config


REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = REPO_ROOT / "docs" / "config_reference.md"


def _leaf_paths(value, prefix=""):
    paths = set()
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, dict):
            paths.update(_leaf_paths(item, path))
        else:
            paths.add(path)
    return paths


class TestConfigReference(unittest.TestCase):
    def test_mentions_every_base_config_field(self):
        reference = REFERENCE_PATH.read_text(encoding="utf-8")
        current_reference = reference.split("## Findings from the old runners", 1)[0]
        config_fields = _leaf_paths(config_to_dict(get_base_config()))
        missing = sorted(
            field for field in config_fields if f"`{field}`" not in current_reference
        )

        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
