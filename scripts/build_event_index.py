"""
Script to aggregate event_{id}.json files from test-trifetch into a single JSON index.

Output files:
- test-trifetch/all_events.json       # list of all events with category and patient_id
- test-trifetch/{category}_events.json  # per-category lists

Usage:
    python scripts/build_event_index.py

This script is safe to run multiple times; it will overwrite the output files.
"""
from pathlib import Path
import json


def build_index(base_path: Path = None, write_per_category: bool = True):
    if base_path is None:
        base_path = Path(__file__).parent.parent / "test-trifetch"

    if not base_path.exists():
        raise FileNotFoundError(f"Base path not found: {base_path}")

    all_events = []
    per_category = {}

    # Iterate categories
    for category_folder in sorted(base_path.iterdir()):
        if not category_folder.is_dir():
            continue
        category_name = category_folder.name
        per_category.setdefault(category_name, [])

        for patient_folder in sorted(category_folder.iterdir()):
            if not patient_folder.is_dir():
                continue
            patient_id = patient_folder.name
            event_file = patient_folder / f"event_{patient_id}.json"
            if not event_file.exists():
                # skip if missing
                continue
            try:
                with open(event_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                # skip malformed
                print(f"Warning: failed to read {event_file}: {e}")
                continue

            # Enrich with metadata
            enriched = {
                "category": category_name,
                "patient_folder": patient_id,
                "event_file": str(event_file.relative_to(base_path)),
            }
            # Merge the original event data (do not overwrite metadata keys)
            for k, v in data.items():
                if k not in enriched:
                    enriched[k] = v

            all_events.append(enriched)
            per_category[category_name].append(enriched)

    # Optionally write per-category files
    if write_per_category:
        for cat, items in per_category.items():
            out_file = base_path / f"{cat}_events.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(items, f, indent=2)

    # Write single aggregate file
    out_all = base_path / "all_events.json"
    with open(out_all, "w", encoding="utf-8") as f:
        json.dump(all_events, f, indent=2)

    # Return summary
    summary = {
        "base_path": str(base_path),
        "total_events": len(all_events),
        "categories": {k: len(v) for k, v in per_category.items()},
        "all_events_file": str(out_all),
    }
    return summary


if __name__ == "__main__":
    import sys
    try:
        base = None
        if len(sys.argv) > 1:
            base = Path(sys.argv[1])
        s = build_index(base_path=base)
        print("Aggregation complete:")
        print(json.dumps(s, indent=2))
    except Exception as exc:
        print(f"Error: {exc}")
        raise
