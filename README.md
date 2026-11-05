# LivingMemory

**Version 1.0.0**

*Warning, this code has gone through very little testing as of this point*
*Claude code was used to write this README file*

A tag-based, local-AI memory system for Ollama-backed LLM conversations. LivingMemory stores conversation turns and arbitrary memories in SQLite and lets a local model compress them into AI-filtered summaries — so your chatbot can remember more without ever blowing the context window.

---

## How it works

Every memory row belongs to a **table** (one per user/session) and a **tag** (topic label). Raw conversation turns are stored under the `"conv"` tag. When a table grows too large, call `flushConv()` to summarize the conversation into topic tags, or `cleanTag()` / `cleanAll()` to re-summarize existing tags. Then, if you want to read those
memories later, you can call `rMem()` and feed it the **tag** they were stored under.

Progress and warnings are reported through Python's `logging` module (`logging.getLogger("LivingMemory")`). To see the progress messages from `flushConv` / `cleanTag` / `cleanAll`, turn on `INFO` logging:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

---

## Requirements

- Python 3.9+
- [`ollama`](https://pypi.org/project/ollama/) Python package (installed automatically)
- [Ollama](https://ollama.com) running locally
- Optional: `transformers`, only if you want exact token counts (`tokenizerModel`)

---

## Installation

Install from source:

```bash
git clone https://github.com/DarkTardigrade/LivingMemory.git
cd LivingMemory
pip install .                 # token estimate, no transformers
pip install ".[tokenizer]"    # also installs transformers for exact token counts
```

> **Not yet published to PyPI** — `pip install livingmemory` will **not** work. Install from source as shown above.

---

## Quick start

```python
from LivingMemory import LivingMemory

mem = LivingMemory(model="qwen3:14b", contextWindow=8192)  # model is optional but strongly recommended

# Store a conversation turn
mem.addConv("alice", "user",      "What is the capital of France?")
mem.addConv("alice", "assistant", "Paris.")

# Read back the last 10 turns as ollama.chat messages
messages = mem.rConv("alice", latest=10)

# Ask the AI something using the conversation history
reply = mem.useAI(messages + [{"role": "user", "content": "And Germany?"}])
print(reply.message.content)
```

---

## API reference

### Constructor

```python
LivingMemory(dbPath: str = "livingMemory.db", model: str = None, tokenizerModel: str = None, contextWindow: int = 2048, timeout: int = 180)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dbPath` | `str` | `"livingMemory.db"` | Path to the SQLite database file |
| `model` | `str` | `None` | Ollama model name (e.g. `"qwen3:14b"`), used to run the summarizer via Ollama. **Strongly recommended, not required** — if omitted, a warning is logged and AI-dependent features (summarization, filtering) fall back to storing text as-is. |
| `tokenizerModel` | `str` | `None` | `None` estimates tokens (characters ÷ 3, CJK characters count as 1) with no extra dependencies. Give a Hugging Face repo id (e.g. `"gpt2"`, `"Qwen/Qwen3-14B"`) to count tokens exactly with that tokenizer instead — this needs `transformers` (`pip install ".[tokenizer]"`) and loads the tokenizer once at construction. If it isn't a resolvable repo id, construction raises immediately. |
| `contextWindow` | `int` | `2048` | The model's context window in tokens. Sent to Ollama as `num_ctx` on every call. Split in half into `inputSize` (the most tokens one filtering / summarizing prompt can be, prompt included) and `outputSize` (the most tokens the model can reply with, sent to Ollama as `num_predict`) |
| `timeout` | `int` | `180` | Ollama client timeout in seconds. Ollama replies are not streamed, so this has to cover the model's **whole** reply — raise it for slow models or big `contextWindow`s |

A `contextWindow` under 1024 logs a warning, since half of it is too small for useful summaries.

#### Table names

Every `table` is a real SQLite table, so its name must:

- only use letters, digits, and underscores, 1–256 characters
- **not start with a digit** (`"123"`), **not be an SQL keyword** (`"order"`, `"select"`, `"table"`, …), and **not start with `sqlite_`**

Any name that breaks these rules raises `ValueError` before anything touches the database. A safe habit is to prefix names, e.g. `"user_alice"`, `"user_123"`.

---

### Conversation

#### `addConv(table: str, role: str = "user", memory: str = "blank", extra: dict = None)`
Store one conversation turn. Valid roles are `"user"`, `"assistant"`, `"system"`, and `"tool"` — anything else raises `ValueError`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `table` | `str` | **required** | Session / user identifier |
| `role` | `str` | `"user"` | Speaker label — must be one of `VALID_ROLES` |
| `memory` | `str` | `"blank"` | The message content. Note the default is the literal text `"blank"`, so always pass content (use `""` for an assistant turn that only has `tool_calls`) |
| `extra` | `dict` | `None` | More Ollama message fields stored with the turn: `{"tool_calls": [...]}` or `{"thinking": "..."}` on an assistant turn, `{"tool_name": "remember"}` on a tool turn, `{"images": [...]}` on a user turn. Plain values or Ollama's own objects both work, so `response.message.tool_calls` can be passed straight in. They come back out of `rConv()` as plain dicts. `role` and `content` always win over the same keys in `extra`. |

```python
mem.addConv("alice", "assistant", "", extra={"tool_calls": [{"function": {"name": "remember", "arguments": {"fact": "likes tea"}}}]})
mem.addConv("alice", "tool", "saved", extra={"tool_name": "remember"})
```

---

#### `rConv(table: str, latest: int = 0) -> list`
Return conversation turns for `table` as message dicts ready for `ollama.chat` (including any `extra` fields). Ordered oldest-first. `latest=0` returns all turns.

**Returns:** `list[dict]`

---

#### `flushConv(table: str, tags: list, keep: int = 0)`
Summarize the current conversation into topic tags, then trim the raw `"conv"` entries. Does nothing if `conv` is empty (or if every row would be kept by `keep`).

The stored turns are first turned into a readable transcript for the model: `role: content` lines, `assistant used <tool>: {arguments}` for tool calls, and `[<tool_name> result]: content` for tool turns. Each tag then gets its own `tagMem` pass over that transcript, using the tag's `desc` as the filter.

If the instance has no `model`, nothing is filtered — **the whole transcript is copied into every tag** in `tags`.

**All or nothing:** each tag is filtered and written in turn, and the old `"conv"` rows are only deleted once every tag has finished. If it fails partway (say Ollama dies on the 3rd of 5 tags), the memories already written for tags 1 and 2 are removed again, so the database is left as it was and you can just run it again. The cost: the work already done for tags 1 and 2 is redone on the retry.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `table` | `str` | **required** | Session / user identifier |
| `tags` | `list[str]` | **required** | Topic labels to summarize into (e.g. `["physics", "personal info"]`) |
| `keep` | `int` | `0` | Number of most-recent `"conv"` rows to leave in `"conv"`, so the model still has recent context. They aren't filed this time; they get filed by a later flush. `0` files and deletes all |

---

### Memory compression

#### `cleanTag(table: str, tag: str, model: str = None, maxTok: int = 0, chopTime: int = 9, prompt: str = None)`
Re-compress all memories under `tag`.

`model` here is the **target** model you're compressing for. It's separate from the instance's own `model`, which is what actually runs the summarization. If `model` is `None`, `maxTok` is `0`, or the instance has no `model` configured (no summarizer available), this runs a single AI filter pass via `tagMem` and replaces the originals instead of attempting compression. If the AI call fails, original rows are restored.

Otherwise, it repeatedly compresses the tag until its token count falls below `maxTok`. If the text plus the prompt doesn't fit in `inputSize` (see `contextWindow`), it's first split via `chunkText` and each piece is summarized separately before being re-checked. If the model returns an empty summary, the old memories are kept (an empty piece is kept as it was) rather than wiped.

After `chopTime` loops, falls back to deleting the oldest rows until the token target is met. The last row needed is **cut down** instead of deleted — its oldest text is removed and its newest text kept — so a tag that's a single big row isn't wiped completely.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `table` | `str` | **required** | Session / user identifier |
| `tag` | `str` | **required** | Tag to compress |
| `model` | `str` | `None` | Gates whether the compression loop runs; if `None`, skips it |
| `maxTok` | `int` | `0` | Token target; if `0`, skips the compression loop |
| `chopTime` | `int` | `9` | Compression loops before falling back to chopping the oldest text |
| `prompt` | `str` | `None` | Custom prompt template for the compression loop; supports `{memories}`, `{desc}`, `{memTok}`, `{maxTok}`, `{percentage}` placeholders. Not used by the single filter pass (that always uses `tagMem`'s built-in prompt) |

---

#### `cleanAll(table: str, drop: list = None, model: str = None, maxTok: int = 0, chopTime: int = 9, prompt: str = None)`
Run `cleanTag` on every tag in `table`. All extra parameters are forwarded to `cleanTag`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `table` | `str` | **required** | Session / user identifier |
| `drop` | `list[str]` | `None` | Tags to skip (e.g. `["conv"]`) |
| `model` | `str` | `None` | Forwarded to `cleanTag` |
| `maxTok` | `int` | `0` | Forwarded to `cleanTag` |
| `chopTime` | `int` | `9` | Forwarded to `cleanTag` |
| `prompt` | `str` | `None` | Forwarded to `cleanTag` |

---

#### `tagMem(table: str, tag: str, memories: str, desc: str = None, prompt: str = None)`
AI-filter `memories` by `tag` and write results to `table`. Used internally by `flushConv` and `cleanTag`.

Pass `memories=""` to just create the tag (with its `desc`) and no memories in it.

If the instance has no `model` configured, filtering is skipped entirely — `memories` is written back as-is and a warning is logged. Otherwise, memories are split on `"\n\n"` and grouped into chunks that fit in `inputSize` (see `contextWindow`) after the prompt's own size is subtracted; any single entry bigger than that budget is split via `chunkText` to the same budget first.

Every chunk is filtered **before** anything is written, so if Ollama fails partway, nothing is written (and a retry doesn't duplicate the first half). Chunks where the AI returns `"Blank Entry"` or an empty reply are not written.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `table` | `str` | **required** | Session / user identifier |
| `tag` | `str` | **required** | Topic filter label |
| `memories` | `str` | **required** | Raw memory text to filter (plain string, entries separated by `"\n\n"`) |
| `desc` | `str` | `None` | Filter description for the AI; falls back to `getDesc` if not provided |
| `prompt` | `str` | `None` | Custom prompt template; use `{memories}` and `{desc}` as placeholders. Uses built-in prompt if not provided. |

---

#### `setTagDesc(table: str, tag: str, desc: str)`
Update the `desc` field on every existing row for `tag`. Does nothing if the tag has no rows yet — to create a tag with just a description, call `writeMem(table, tag, desc)` with no memory first.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `table` | `str` | **required** | Session / user identifier |
| `tag` | `str` | **required** | Tag whose desc to update |
| `desc` | `str` | **required** | New description string |

---

#### `getDesc(table: str, tag: str) -> str`
Return the `desc` stored for `tag`. Falls back to the tag key string if the tag has no rows.

**Returns:** `str`

---

### Low-level read / write / delete

#### `writeMem(table: str, tag: str = "all", desc: str = "all", memory: str = "", time = None)`
Write one memory row to the database. Creates the table if it does not exist yet. `desc="all"` is a sentinel meaning "use tag as desc"; `time=None` uses the current timestamp.

With no `memory`, the row just makes the tag exist and holds its `desc`. `rMem` skips empty rows, so they never end up in a summary.

```python
mem.writeMem("alice", "projects", "Projects Alice is working on and how far along they are")
```

---

#### `readMem(table: str, tag: str = "all", latest: int = 0) -> list`
Return raw SQLite row tuples. Each row is `(key, tag, desc, memory, time)` — memory text is at index `[3]`. `tag="all"` returns every tag; `latest=0` returns all rows. Empty rows are included.

**Returns:** `list[tuple]` — returns `[]` if the table does not exist.

---

#### `rMem(table: str, tag: str = "all", latest: int = 0, time: bool = False) -> str`
Same as `readMem`, but returns all matching memory texts joined into a single string by `"\n\n"`, leaving out empty rows. If `time=True`, each block is prefixed with its timestamp on its own line.

---

#### `deleteTag(table: str, tag: str, keep: int = 0)`
Delete rows under `tag`. `keep=0` deletes all; `keep=N` preserves the newest N rows. Does nothing if the table does not exist.

---

### Utilities

#### `useAI(messages: list, tools: list = None, model: str = None)`
Send an `ollama.chat`-style message list to the model and return the raw Ollama response object. `model` overrides the instance default for this call. The instance's `contextWindow` is sent as `num_ctx`, so Ollama runs the model with the same context window LivingMemory plans for.

**Returns:** raw `ollama.Response` (access reply via `.message.content`).  
**Raises:** `ValueError` if no `model` is given and the instance has none configured; `RuntimeError` on any Ollama or connection error. Errors are also emitted via `logging.getLogger(__name__)` before raising.

---

#### `getTags(table: str, drop: list = None) -> list`
List all distinct tag labels in a table.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `table` | `str` | **required** | Session / user identifier |
| `drop` | `list[str]` | `None` | Tags to exclude from results (e.g. `["conv"]`) |

**Returns:** `list[str]`. Returns `[]` if the table does not exist.

---

#### `memRows(table: str, tag: str = "all") -> int`
Count the number of memory rows matching the tag.

---

#### `memTok(table: str, tag: str = "all", model: str = None) -> int`
Count the total number of tokens across all matching memories (estimated, or exact if `tokenizerModel` was given). Empty rows are skipped, and the `"\n\n"` between memories is counted too. `model` is unused and kept so older calls still work.

---

#### `chunkText(text: str, model: str = None, CW: int = None) -> list`
Split `text` into pieces that each fit within a token budget. Returns `[text]` unchanged if it already fits. For oversized text, estimates a conservative characters-per-token ratio (with a built-in safety margin) and walks the string in fixed-size steps until it's fully covered, so no trailing text is ever dropped — chunks may split mid-word/mid-token. Used internally by `cleanTag` and `tagMem`, and safe to call directly.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | **required** | Text to split |
| `model` | `str` | `None` | Unused, kept so older calls still work |
| `CW` | `int` | `None` | Token budget per chunk; falls back to `inputSize` (half of `contextWindow`) |

**Returns:** `list[str]`

---

#### `LivingMemory.now() -> str`
Return the current timestamp as `"YYYY-MM-DD // HH:MM:SS"`. Static method.

---

### Internal methods

#### `_quickPrompt(prompt: str, model: str = None) -> str`
Send a single prompt to the model with no history and return the response as a plain string. `model` overrides the instance default. Raises `RuntimeError` if Ollama is unreachable.

---

#### `_strTok(text: str) -> int`
Count the tokens in a string. Uses the `tokenizerModel` tokenizer if one was given at construction, otherwise estimates (characters ÷ 3, rounded up; CJK characters count as 1 token each). Needs no `model`.

---

#### `LivingMemory._validateTable(table: str)`
Raise `ValueError` if `table` isn't a usable table name (see [Table names](#table-names)). It first checks `[A-Za-z0-9_]{1,256}`, then has SQLite try the name on a throwaway in-memory database, so names like `"123"`, `"order"`, or `"sqlite_x"` are rejected the same way SQLite would.

---

#### `LivingMemory._tableExists(cursor, table: str) -> bool`
Check whether a table already exists in the connected SQLite database.

---

## Example: flushing conversation memory into topic summaries

```python
from LivingMemory import LivingMemory

mem = LivingMemory(model="qwen3:14b", contextWindow=32768)

# ... after many conversation turns ...

print(mem.memRows("alice"))   # total row count
print(mem.memTok("alice"))    # total token count

# Summarize into topics, keeping the 5 most-recent raw turns
mem.flushConv(
    table="alice",
    tags=["personal preferences", "ongoing projects"],
    keep=5,
)

# Read a topic summary back
summary = mem.rMem("alice", tag="ongoing projects")
```

**Tip:** giving `flushConv()` a tag that doesn't exist yet creates it.
**Tip:** to flush into "every tag except a few", pass `tags=mem.getTags(table, drop=[...])` (only works once those tags already have content).
**Tip:** to give a new tag a better filter description before its first flush, create it with `writeMem(table, tag, desc)`.

---

## Database schema

Each session table becomes its own SQLite table:

```
key       INTEGER  PRIMARY KEY AUTOINCREMENT
tag       TEXT     topic label (e.g. "conv", custom topic strings)
desc      TEXT     longer description used in AI filter prompts (defaults to tag)
memory    TEXT     the memory content (empty = a tag with only a description)
time      TEXT     timestamp written at insert time ("YYYY-MM-DD // HH:MM:SS")
```

Rows are read back sorted by `time`, then `key`. The default timestamp format sorts correctly as text; if you pass your own `time` to `writeMem`, use the same format or the order will be wrong.

`keep` in `flushConv` / `deleteTag` and `latest` in `readMem` / `rConv` count rows in that same order.

---

## License

This project is licensed under the GNU General Public License v3.0 or later (`GPL-3.0-or-later`).
See the [LICENSE](LICENSE) file for details.