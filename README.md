<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="img/logo-dark.png">
    <img alt="logo" src="img/logo-light.png" width="500px">
  </picture>
  <br>
  Use LLMs for document ranking.
</p>

## Description

Got a bunch of data? Want to throw it at an LLM to find the most "interesting" stuff? If you simply YOLO your data into a ChatGPT session, you'll run into problems:
- Nondeterminism: Doesn't always respond with the same result
- Limited context: Can't pass in all the data at once, need to break it up
- Output contraints: Sometimes doesn't return all the data you asked it to review
- Scoring subjectivity: Struggles to assign a consistent numeric score to an individual item

`siftrank` is an implementation of the **Sift**Rank document ranking algorithm that uses LLMs to efficiently find the items in any dataset that are most relevant to a given prompt:
- **S**tochastic: Randomly samples the dataset into small batches.
- **I**nflective: Looks for a natural inflection point in the scores that distinguishes particularly relevant items from the rest.
- **F**ixed: Caps the maximum number of LLM calls so the computational complexity remains linear in the worst case.
- **T**rial: Repeatedly compares batched items until the relevance scores stabilize.

Use LLMs to rank anything. No fine-tuning. No domain-specific models. Just an off-the-shelf model and your ranking prompt. Typically runs in seconds and costs pennies.

## Getting started

### Install

```
go install github.com/noperator/siftrank/cmd/siftrank@latest
```

### Configure

Set your `OPENAI_API_KEY` environment variable.

### Usage

```
siftrank -h

Options:
  -f, --file string     input file (required)
  -m, --model string    OpenAI model name (default "gpt-4o-mini")
  -o, --output string   JSON output file
  -p, --prompt string   initial prompt (prefix with @ to use a file)
  -r, --relevance       post-process each item by providing relevance justification (skips round 1)

Visualization:
      --no-minimap   disable minimap panel in watch mode
      --watch        enable live terminal visualization (logs suppressed unless --log is specified)

Debug:
  -d, --debug          enable debug logging
      --dry-run        log API calls without making them
      --log string     write logs to file instead of stderr
      --trace string   trace file path for streaming trial execution state (JSON Lines format)

Advanced:
  -u, --base-url string         OpenAI API base URL (for compatible APIs like vLLM)
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

Compares 100 [sentences](https://github.com/noperator/siftrank/blob/main/testdata/sentences.txt) in 7 seconds.

```
siftrank \
    -f testdata/sentences.txt \
    -p 'Rank each of these items according to their relevancy to the concept of "time".' |
    jq -r '.[:10] | map(.value)[]' |
    nl

   1  The train arrived exactly on time.
   2  The old clock chimed twelve times.
   3  The clock ticked steadily on the wall.
   4  The bell rang, signaling the end of class.
   5  The rooster crowed at the break of dawn.
   6  She climbed to the top of the hill to watch the sunset.
   7  He watched as the leaves fell one by one.
   8  The stars twinkled brightly in the clear night sky.
   9  He spotted a shooting star while stargazing.
  10  She opened the curtains to let in the morning light.
```

<details><summary>Advanced usage</summary>

#### JSON support

If the input file is a JSON document, it will be read as an array of objects and each object will be used for ranking.

For instance, two objects would be loaded and ranked from this document:

```json
[
  {
    "path": "/foo",
    "code": "bar"
  },
  {
    "path": "/baz",
    "code": "nope"
  }
]
```

#### Templates

It is possible to include each element from the input file in a template using the [Go template syntax](https://pkg.go.dev/text/template) via the `--template "template string"` (or `--template @file.tpl`) argument.

For text input files, each line can be referenced in the template with the `Data` variable:

```
Anything you want with {{ .Data }}
```

For JSON input files, each object in the array can be referenced directly. For instance, elements of the previous JSON example can be referenced in the template code like so:

```
# {{ .path }}

{{ .code }}
```

Note in the following example that the resulting `value` key contains the actual value being presented for ranking (as described by the template), while the `object` key contains the entire original object from the input file for easy reference.

```
# Create some test JSON data.
seq 9 |
    paste -d @ - - - |
    parallel 'echo {} | tr @ "\n" | jo -a | jo nums=:/dev/stdin' |
    jo -a |
    tee input.json

[{"nums":[1,2,3]},{"nums":[4,5,6]},{"nums":[7,8,9]}]

# Use template to extract the first element of the nums array in each input object.
siftrank \
	-f input.json \
	-p 'Which is biggest?' \
	--template '{{ index .nums 0 }}' \
	--max-trials 1 |
	jq -c '.[]'

{"key":"eQJpm-Qs","value":"7","object":{"nums":[7,8,9]},"score":0,"exposure":1,"rank":1}
{"key":"SyJ3d9Td","value":"4","object":{"nums":[4,5,6]},"score":2,"exposure":1,"rank":2}
{"key":"a4ayc_80","value":"1","object":{"nums":[1,2,3]},"score":3,"exposure":1,"rank":3}
```

</details>

## Back matter

### Acknowledgements

I released the prototype of this tool, Raink, while at Bishop Fox. See the original [presentation](https://www.youtube.com/watch?v=IBuL1zY69tY), [blog post](https://bishopfox.com/blog/raink-llms-document-ranking) and [CLI tool](https://github.com/bishopfox/raink).

### See also

- [O(N) the Money: Scaling Vulnerability Research with LLMs](https://noperator.dev/posts/on-the-money/)
- [Using LLMs to solve security problems](https://noperator.dev/posts/ai-for-security/)
- [Hard problems that reduce to document ranking](https://noperator.dev/posts/document-ranking-for-complex-problems/)
- [Commentary: Critical Thinking - Bug Bounty Podcast](https://youtu.be/qd08UBNpu7k?si=pMVEYtmKnyuJkL9B&t=1511)
- [Discussion: Hacker News](https://news.ycombinator.com/item?id=43174910)
- [Large Language Models are Effective Text Rankers with Pairwise Ranking Prompting](https://arxiv.org/html/2306.17563v2)

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

### License

This project is licensed under the [MIT License](LICENSE).
