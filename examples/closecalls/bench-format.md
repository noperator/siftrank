# recorded runs in the ranking review bench

export a retained run into the bench's existing generic import format:

```sh
python3 examples/closecalls/bench_export.py \
  --results examples/closecalls/network/rankings.json \
  --fixture examples/closecalls/network/input.json \
  --case ctu-45 --seed 1 --mode pairwise \
  --comparisons examples/closecalls/network/captures.jsonl.gz \
  --output /tmp/siftrank-review-example
```

choose a new output directory. this uses the included recorded network run and
makes no provider calls. the selector is explicit: case, seed and ranking mode.

- `results.json` is a native ranked-document array. import it as results.
- `input.json` is the original candidate value strings in source order. attach it
  as original input. separate relevance labels are never included.
- `comparisons.jsonl` retains the selected run's original comparison receipt
  lines. source identities and text are checked against the input. this file is
  separate from trial replay; the bench does not currently display pair votes.
- `manifest.json` records source hashes, stable key/index mappings, selection,
  and missing-output coverage. it is a receipt, not a bench import envelope.

score, exposure, and rounds remain the exact recorded fields. no confidence
score is inferred. rows keep `document: null` because the engine did not store an
embedded original. the exporter verifies each key, value and input position;
within the bench, an attached original remains labeled as a position match.

all source candidates survive in `input.json`, including any absent from the
ranked output. this measures coverage of the supplied fixture, not the full
traffic corpus before candidate selection. that earlier total remains unknown
in the export.

if actual trial snapshots exist, `--trace path/to/trace.jsonl` validates and
copies them. the included run did not record those snapshots, so no trace is
created. comparisons and final scores cannot reconstruct the missing history.
when `run_id` exists, comparison receipts are selected by that recorded id;
otherwise the caller must supply a single matching run's receipts. trace run
attribution is also caller supplied; matching source ids alone cannot prove it.

## current integration boundary

the actual bench importer accepted all 64 rows from this run with no issues,
matched all 64 input positions, and preserved identities when the same bytes
were imported again. its review export retained a selected item's note and
shortlist identity. this was an importer/exporter check, not browser interaction
or a provider execution test.

the current bench cannot import its own review or shortlist export envelopes,
and does not have a comparison inspector. those remain separate integration
work. importing this example neither reconnects a provider nor adds those
features.
