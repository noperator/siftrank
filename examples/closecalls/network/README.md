# captured-traffic replay

an offline experiment using 320 host/time windows from three recorded network captures,
with publisher-supplied normal/botnet labels. all six ordinal pairwise runs are included:
two seeds per capture, 498 requests and 13,062 pair observations. no new inference is needed.

from the repository root:

```sh
python3 examples/closecalls/network_review.py --output /tmp/network-review.json
python3 -m unittest discover -s examples/closecalls -p 'test_network_review.py'
```

the replay verifies bundled hashes, source identities, batch context, request state,
probability orientation, wire responses and token counts before opening the evaluation
labels. `input.json` and `captures.jsonl.gz` retain the actual normalized candidate content;
`wire.jsonl.gz` retains full recorded requests and responses. `labels.json` is separate from
model input. `rankings.json` retains the final document scores, exposures, rounds and ranks.

at 20 unique pair reviews per run, closest mean preferences identify **2.17 wrong final
orderings on average**, compared with **0.76 expected from random review**. closest individual
preferences find 1.50; strong reversals first find 1.67. see `results.json` for every run and
the complete curves. queues use only model observations and stable identity hashes;
equal-grade and unknown pairs still spend review budget.

this uses a different unit from the neighboring synthetic example: unique pairs and errors
in the final ordering, rather than repeated observations and errors in individual preferences.
the two results should not be pooled.

the cohort is restricted to publisher-labeled normal/botnet flows. after that restriction,
window selection retains 320 of 448 candidates and 96 of 163 known-positive windows, losing
67 positive windows and four host/behavior groups before ranking. `provenance.json` records
the source hashes, selected runs, transformation and per-capture coverage. source aggregation
retains totals but only three representative flow records per window; original full traffic
files remain external.

these are retrospective 2011 captures from controlled malware experiments, not live production
traffic. repeated seeds and related windows are correlated. wrong relative order is a proxy;
these results do not establish recovered incidents, causal links or analyst time saved.

## source and attribution

the [source dataset](https://www.stratosphereips.org/datasets-ctu13) is credited to
sebastian garcia, martin grill, jan stiborek and alejandro zunino, *an empirical comparison of
botnet detection methods*, computers & security 45 (2014), 100–123,
[doi:10.1016/j.cose.2014.05.011](https://doi.org/10.1016/j.cose.2014.05.011).
the publisher's [dataset overview](https://www.stratosphereips.org/datasets-overview)
states that this dataset is available under [cc-by 2.0](https://creativecommons.org/licenses/by/2.0/).
this derived package hashes host addresses,
aggregates flows, selects windows and adds model comparisons. those changes are ours;
the original authors do not endorse the experiment. original source names remain in machine
provenance so the files can be identified unambiguously.
