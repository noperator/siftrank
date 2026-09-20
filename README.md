<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="img/logo-dark.png">
    <img alt="SiftRank" src="img/logo-light.png" width="500px">
  </picture>
  <br>
  Use LLMs to find the needles in your haystack.
</p>

**19 Sep '26: SiftRank now supports Jev**, alongside OpenAI and compatible Chat Completions APIs. Use `--provider jev` to get started; see [Jev](#jev) below.

**Using an agent?** Install the [SiftRank agent skill](#agent-skill) to help it retrieve the most relevant items from large datasets on the fly.

## Description

You've got a *lot* of data on your hands, and you need to find signal in the noise. Problem is, your desired "signal" doesn't fit a predefined regex (more of a semantic you-know-it-when-you-see-it kind of thing). And there's way more stuff to look at than you can possibly read through. Even your agent's context window can't fit it all! *Has this ever happened to you?*

Tell SiftRank what you're looking for, and it quickly searches your data to bring the most relevant items to the top. Rank prose passages, files, JSON, search results, code, support tickets, product listings, or anything else you can represent as text.

If you otherwise simply YOLO your data into a ChatGPT session and ask it to find what matters, you'll run into problems:

- **Nondeterminism:** Ask again, and you may get a different answer.
- **Limited context:** Your entire collection may not fit in one request.
- **Output constraints:** The model may omit items or stop before finishing the list.
- **Scoring subjectivity:** A numeric score assigned to one item may not be comparable to a score assigned in another request.

SiftRank breaks the collection into small, randomized batches and asks a model to compare items against your ranking prompt. It combines those relative orderings across repeated trials, checks for convergence, and progressively focuses on the most relevant items. You get a ranked list, with the greatest precision concentrated near the top. Use an off-the-shelf model and a prompt describing what you want to find. No fine-tuning or domain-specific model required. Small, fast models work great.

For the algorithm and an application to vulnerability research, see [Sift or Get Off the PoC: Applying Information Retrieval to Vulnerability Research with SiftRank](https://arxiv.org/abs/2512.06155). For another practical example, see VulnCheck's [SiftRanking Canary Intelligence](https://www.vulncheck.com/blog/siftrank_canaries), which uses SiftRank to prioritize exploitation telemetry for investigation.

## Getting started

### Install

```sh
go install github.com/noperator/siftrank/cmd/siftrank@latest
```

Note that installing the CLI with `go install` does not install the skill.

### Agent skill

The [SiftRank agent skill](.agents/skills/siftrank/SKILL.md) teaches an agent how to decompose a large collection of data into useful candidates to be ranked, choose how to format the input data, run SiftRank, and investigate the strongest results in their original context. Use it for *any* task where there is too much material to read directly.

The skill uses the portable Agent Skills format. Install the `siftrank` folder in your agent's supported skills directory:

```sh
git clone https://github.com/noperator/siftrank
cd siftrank
mkdir -p ~/.agents/skills
cp -R .agents/skills/siftrank ~/.agents/skills/
```

Install the CLI and configure an API key as described below, then ask your agent to use the skill. For example:

> Use the SiftRank skill to find the Jira comments in this data export that best explain why the project was delayed. Preserve source locations and inspect the highest-ranked passages before summarizing your findings.

### Configure

OpenAI is the default provider. Set `OPENAI_API_KEY`, or store your settings in a configuration profile in `~/.config/siftrank/config.yaml` or `./config.yaml`.

```yaml
default: nano
profiles:
  nano:
    provider: openai
    api_key_cmd: op read op://myvault/openai-api-key/credential
    model: gpt-5-nano-2025-08-07
    effort: minimal
```

This example retrieves the key with the 1Password CLI. Use `api_key_cmd` to retrieve a secret from a command, or `api_key` to supply it directly.

The profile selected by `default` loads when `--profile` is not specified. CLI flags override profile settings; the provider's API key environment variable takes precedence over the profile's key. See [`config-example.yaml`](config-example.yaml) for all available options.

To use another compatible Chat Completions API, set `--base-url` to its API URL, including `/v1`, and `--model` to a model it supports. Select a model and endpoint that support the structured output SiftRank requests.

### Usage

```
Use LLMs for document ranking via the SiftRank algorithm

Usage:
  siftrank [flags]

Options:
      --config-file string   path to config file (overrides discovery)
  -f, --file string          input file (required)
  -m, --model string         model name (Jev default: jev-latest) (default "gpt-4o-mini")
  -o, --output string        JSON output file
  -P, --profile string       use a named profile from the config file
  -p, --prompt string        initial prompt (prefix with @ to use a file)
      --provider string      ranking provider: openai or jev (default "openai")
  -r, --relevance            post-process each item by providing relevance justification (skips round 1)

Visualization:
      --no-minimap   disable minimap panel in watch mode
      --watch        enable live terminal visualization (logs suppressed unless --log is specified)

Debug:
  -d, --debug          enable debug logging
      --dry-run        log API calls without making them
      --log string     write logs to file instead of stderr
      --trace string   trace file path for streaming trial execution state (JSON Lines format)

Advanced:
  -u, --base-url string         provider API base URL, including /v1
  -b, --batch-size int          number of items per batch (default 10)
  -c, --concurrency int         max concurrent LLM calls across all trials (default 50)
  -e, --effort string           reasoning effort level: none, minimal, low, medium, high
      --elbow-method string     elbow detection method: curvature (default), perpendicular (default "curvature")
      --elbow-tolerance float   elbow position tolerance (0.05 = 5%) (default 0.05)
      --encoding string         tokenizer encoding (default "o200k_base")
      --json                    force JSON parsing regardless of file extension
      --max-trials int          maximum number of ranking trials (default 50)
      --min-trials int          minimum trials before checking convergence (default 5)
      --no-converge             disable early stopping based on convergence
      --ratio float             refinement ratio (0.0-1.0, e.g. 0.5 = top 50%) (default 0.5)
      --stable-trials int       stable trials required for convergence (default 5)
      --template string         template for each object (prefix with @ to use a file) (default "{{.Data}}")
      --tokens int              max tokens per batch (default 128000)

Flags:
  -h, --help   help for siftrank
```

#### Rank text

For plain text input, each nonempty line is one item. From a checkout of this repository, try ranking the [sample sentences](testdata/sentences.txt):

```sh
siftrank \
  --file testdata/sentences.txt \
  --prompt 'Rank these sentences by relevance to time, clocks, and the passage of time.' \
  > ranked.json

jq -r '.[:10][].value' ranked.json | nl
```

The output is a JSON array ordered from most to least relevant. `rank: 1` is the top result. The `score` is an internal ranking score, not a probability or a calibrated confidence value.

Add `--watch` for a live terminal visualization.

#### Jev

SiftRank asks Jev to compare pairs of items within each batch, then combines the returned probabilities into an ordering and runs its normal ranking algorithm. Set `TYPESAFE_API_KEY` and select Jev on the command line:

```sh
export TYPESAFE_API_KEY='your-api-key'

siftrank \
  --provider jev \
  --model jev-latest \
  --effort '' \
  --file testdata/sentences.txt \
  --prompt 'Rank these sentences by relevance to time, clocks, and the passage of time.' \
  > ranked.json
```

The default model is `jev-latest` when no model is supplied by a flag or profile. The command above explicitly selects it and clears any reasoning effort inherited from a default profile. Jev does not support reasoning effort or the `--relevance` option.

For an experimental example that retains pair probabilities and tests whether close decisions help find ranking mistakes, see [close-call review](examples/closecalls).

#### Rank JSON objects with a template

For structured data, pass a JSON array and a [Go template](https://pkg.go.dev/text/template) describing what the model should see. This lets you control the ranking input while preserving each complete original object for downstream use.

For example, save this as `candidates.json`:

```json
[
  {
    "id": "article-1",
    "title": "Repairing a mechanical clock",
    "content": {
      "excerpt": "How to diagnose a worn escapement and restore accurate timekeeping."
    },
    "source": {
      "url": "https://example.com/clock-repair"
    },
    "metadata": {
      "collection": "saved-articles"
    }
  },
  {
    "id": "article-2",
    "title": "Growing tomatoes in containers",
    "content": {
      "excerpt": "Choosing soil, watering consistently, and supporting tomato plants."
    },
    "source": {
      "url": "https://example.com/container-tomatoes"
    },
    "metadata": {
      "collection": "saved-articles"
    }
  }
]
```

Save this as `candidate.tmpl`:

```gotemplate
ID: {{ .id }}
Title: {{ .title }}
Source: {{ .source.url }}

{{ .content.excerpt }}
```

Then rank the objects:

```sh
siftrank \
  --file candidates.json \
  --template @candidate.tmpl \
  --prompt 'Rank these articles by how useful they would be to someone repairing a clock.' \
  > ranked.json
```

Each result's `value` contains the rendered template, and `document` contains the entire original object, including `metadata`, which this template does not show to the model. To extract the top 20 original objects:

```sh
jq '[.[:20][].document]' ranked.json
```

Templates can also be supplied inline with `--template '{{ .title }}: {{ .content.excerpt }}'`. Use `{{ index .tags 0 }}` to access an array element. For plain text input, use `{{ .Data }}` to reference the line.

Include a unique ID or source locator in the template when otherwise identical text represents distinct items. SiftRank derives item keys from the rendered text.

#### Input and output details

- Files ending in `.json` are read as JSON arrays. Use `--json` to force JSON parsing for another filename or a stream. JSON Lines is not the same format.
- On Unix, read standard input with `--file /dev/stdin`; add `--json` when piping a JSON array.
- Use `--prompt @prompt.txt` or `--template @candidate.tmpl` to load longer prompts or templates from files.
- Results include the rendered `value`, original JSON `document` (or `null` for text), `rank`, `score`, `exposure`, `rounds`, and a zero-based `input_index` referring to the original input.
- SiftRank adjusts batch size to fit estimated provider limits. It does not split oversized individual items; split those before ranking.

#### Tips

Each input item is something you want ranked. Split large data sources into useful pieces: sections of a report, passages from a book, individual messages, functions in a codebase, or whole files when they are small enough.

Smaller items let the model make more focused comparisons, but they should have enough context to stand on their own. Preserve titles, surrounding explanations, and source locations where useful. A JSON object (shown above) can retain additional metadata without including it all in the template.

Write a prompt that states the actual selection criterion: "most useful for diagnosing intermittent connection failures" is more specific than "how to fix my network." Inspect the leading results and follow their source references when you need more context.

## Back matter

### See also

- **Phrack article**: link will be added when published
- **Black Hat USA talk**: link will be added when published
- [Sift or Get Off the PoC: Applying Information Retrieval to Vulnerability Research with SiftRank](https://arxiv.org/abs/2512.06155)
- [SiftRanking Canary Intelligence](https://www.vulncheck.com/blog/siftrank_canaries)
- [O(N) the Money: Scaling Vulnerability Research with LLMs](https://noperator.dev/posts/on-the-money/)
- [Using LLMs to solve security problems](https://noperator.dev/posts/ai-for-security/)
- [Hard problems that reduce to document ranking](https://noperator.dev/posts/document-ranking-for-complex-problems/)
- [Commentary: Critical Thinking - Bug Bounty Podcast](https://youtu.be/qd08UBNpu7k?si=pMVEYtmKnyuJkL9B&t=1511)
- [Discussion: Hacker News](https://news.ycombinator.com/item?id=43174910)
- [Large Language Models are Effective Text Rankers with Pairwise Ranking Prompting](https://arxiv.org/html/2306.17563v2)

### Acknowledgements

I released the prototype of this tool, Raink, while at Bishop Fox. See the original [presentation](https://www.youtube.com/watch?v=IBuL1zY69tY), [blog post](https://bishopfox.com/blog/raink-llms-document-ranking), and [CLI tool](https://github.com/bishopfox/raink).

### To-do

- [ ] add python bindings?
- [ ] allow specifying an input _directory_ (where each file is distinct object)
- [ ] clarify when prompt included in token estimate
- [ ] factor LLM calls out into a separate package
- [ ] run openai batch mode
- [ ] report cost + token usage
- [ ] add more examples, use cases
- [ ] account for reasoning tokens separately

<details><summary>Completed</summary>

- [x] add visualization
- [x] support reasoning effort
- [x] add blog link
- [x] add parameter for refinement ratio
- [x] add ~boolean~ refinement ratio flag
- [x] alert if the incoming context window is super large
- [x] automatically calculate optimal batch size?
- [x] explore "tournament" sort vs complete exposure each time
- [x] make sure that each randomized run is evenly split into groups so each one gets included/exposed
- [x] parallelize openai calls for each run
- [x] remove token limit threshold? potentially confusing/unnecessary
- [x] save time by using shorter hash ids
- [x] separate package and cli tool
- [x] some batches near the end of a run (9?) are small for some reason
- [x] support non-OpenAI models

</details>

## License

This project is licensed under the [MIT License](LICENSE).
