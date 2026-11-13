# LangChain, Targeted: The Parts That Make LangGraph Easier

A hands-on, runnable path through **only the LangChain you need to understand and use LangGraph well**:
12 lessons, a capstone, 56 offline tests, and the same lesson plan as the sibling
[`learn-langgraph`](../learn-langgraph) tutorial.

This is not a tour of all of LangChain. Chains, retrievers, document loaders and the long tail of
integrations are left out on purpose. Every lesson answers one question a LangGraph learner actually hits:
*what is this object that a node returns? what is `create_agent` really doing? where does this middleware
live in the graph?*

Everything runs **offline by default** (a scripted fake chat model). Add an API key to `.env` and the same
code uses a real model. Verified against `langchain 1.5.0`, `langgraph 1.2.21`, `langchain-core 1.8.0`,
`langchain-anthropic 1.7.10`, Python 3.14 (the project requires Python 3.12+), on 2026-11-12.

## Table of contents

- [How to use this repo](#how-to-use-this-repo)
- [Setup](#setup)
- [Where LangChain ends and LangGraph begins](#where-langchain-ends-and-langgraph-begins)
- [Curriculum at a glance](#curriculum-at-a-glance)
- [Track A: The building blocks (01-05)](#track-a-the-building-blocks-01-05)
  - [Lesson 01: Runnables and chat models](#lesson-01-runnables-and-chat-models)
  - [Lesson 02: Messages](#lesson-02-messages)
  - [Lesson 03: Tools](#lesson-03-tools)
  - [Lesson 04: Structured output](#lesson-04-structured-output)
  - [Lesson 05: Model and tools in a graph](#lesson-05-model-and-tools-in-a-graph)
- [Track B: create_agent is a LangGraph graph (06-09)](#track-b-create_agent-is-a-langgraph-graph-06-09)
  - [Lesson 06: create_agent](#lesson-06-create_agent)
  - [Lesson 07: State, context and memory](#lesson-07-state-context-and-memory)
  - [Lesson 08: Middleware hooks](#lesson-08-middleware-hooks)
  - [Lesson 09: Prebuilt middleware](#lesson-09-prebuilt-middleware)
- [Track C: Interactive agents and composition (10-12)](#track-c-interactive-agents-and-composition-10-12)
  - [Lesson 10: Human-in-the-loop](#lesson-10-human-in-the-loop)
  - [Lesson 11: Streaming](#lesson-11-streaming)
  - [Lesson 12: Multi-agent systems and dropping down](#lesson-12-multi-agent-systems-and-dropping-down)
- [Capstone: Help desk with approval gate](#capstone-help-desk-with-approval-gate)
- [Study plans](#study-plans)
- [Concepts in one page](#concepts-in-one-page)
- [Project layout](#project-layout)
- [Testing and quality gates](#testing-and-quality-gates)
- [What was left out, and why](#what-was-left-out-and-why)
- [Official docs map](#official-docs-map)

## How to use this repo

- **Every lesson is one Python file.** The explanation is in the docstring, the code runs, and each file has
  exercises at the bottom. Read the docstring, run it, change something, break it. The capstone is a
  single file too.
- **Every lesson below has the same lesson plan:** prerequisites, an overview, learning objectives, the
  concepts covered, a walkthrough of what is in the file, how to run it and what you should see, common
  mistakes, exercises, and questions to check your understanding. It is the same plan `learn-langgraph` uses.
- **Objectives are written as observable skills** (describe, build, predict, diagnose, choose). Treat them as
  the exit criteria: if you cannot do them without looking, redo the exercises before moving on.
- **Each check-your-understanding question is tagged `[O#]`** with the objective it tests, mixes recall,
  application and analysis, and has an answer in a collapsible block. Answer first, then compare.
- **Prerequisites are cumulative.** You do not need to finish `learn-langgraph` first, but lessons 05-12 are
  much richer if you have seen its lessons 01-03 (state, reducers, routing). The table in
  [Where LangChain ends and LangGraph begins](#where-langchain-ends-and-langgraph-begins) says which
  `learn-langgraph` lesson each lesson here pairs with.
- **Behavior claims were checked by running them**, and `tests/test_lessons.py` pins them, so an upgrade
  tells you when a lesson has gone stale. Where a default surprised me (a raising tool crashes the run, for
  example) the lesson says so and a test proves it.

## Setup

Windows PowerShell (Python 3.12+; the project builds with Hatch/Hatchling):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"      # runtime deps + pytest, ruff, ty
copy .env.example .env       # optional: add a key to use a real LLM
pytest -q                    # should be all green (56 tests, no network needed)
```

Run any lesson: `python lessons/06_create_agent.py`

With a real model: put `ANTHROPIC_API_KEY=...` in `.env`. Lessons pick the model from
`lessons/common.py` (`DEFAULT_MODEL`), or from `LC_MODEL` (any `init_chat_model` string such as
`anthropic:claude-haiku-5-5`). A real model's wording will differ from the scripted replies shown below,
but the shape of every trace (which messages, which tool calls, which pauses) is the same.

If you use Hatch: `hatch run test`, `hatch run lint`, `hatch run typecheck`, `hatch run check`.

## Where LangChain ends and LangGraph begins

One mental split removes most of the confusion:

| | LangChain (this repo) | LangGraph (`learn-langgraph`) |
|---|---|---|
| **Owns** | The *things* that flow: models, messages, tools, schemas, prompts, agent factory, middleware | The *engine* that runs them: state, nodes, edges, checkpoints, interrupts, streaming, subgraphs |
| **Typical object** | `AIMessage`, `@tool`, `ChatModel`, `create_agent` | `StateGraph`, `add_node`, `Command`, `interrupt` |
| **Relationship** | `create_agent` *returns* a LangGraph graph; middleware hooks *become* its nodes | Runs any Runnable as a node; checkpointers, `thread_id`, `stream_mode` work on LangChain agents unchanged |

| Lesson here | Pairs with `learn-langgraph` | The link |
|---|---|---|
| 01 Runnables | 01 Your first graph | A compiled graph *is* a Runnable (`invoke`, `batch`, `stream`) |
| 02 Messages | 02 State and reducers | `messages` is a state key with the `add_messages` reducer |
| 03 Tools | 05 ReAct from scratch | The tool schema is what the model sees; `ToolNode` runs the calls |
| 04 Structured output | 03 Routing and loops | A typed classifier output drives a conditional edge |
| 05 Model and tools in a graph | 05 and 06 | The whole agent loop from LangChain parts |
| 06 create_agent | 06 Prebuilt pieces | `create_agent` builds that loop; it is a compiled graph |
| 07 State, context, memory | 07 Persistence, 10 Long-term memory | Checkpointer = short-term, Store = long-term, `Runtime` = per-run context |
| 08 Middleware hooks | 03 Routing, 12 Reliability | A node-style hook is a node; `jump_to` is `Command(goto=END)` |
| 09 Prebuilt middleware | 12 Production reliability | Retry, limits, fallback, summarization as one-liners |
| 10 Human-in-the-loop | 08 Human-in-the-loop | `HumanInTheLoopMiddleware` is `interrupt()` for tool approval |
| 11 Streaming | 09 Streaming | Same `stream_mode`s, applied to an agent |
| 12 Multi-agent | 11 Subgraphs and multi-agent | An agent is a subgraph; drop to `StateGraph` for control |
| Capstone | Capstone | Agents as nodes, plus middleware, approval and context |

## Curriculum at a glance

| # | Lesson | Core idea | Why a LangGraph user needs it |
|---|--------|-----------|-------------------------------|
| 01 | [Runnables and models](#lesson-01-runnables-and-chat-models) | One interface: `invoke`/`stream`/`batch`, `\|` | A graph is a Runnable; a node can be a chain |
| 02 | [Messages](#lesson-02-messages) | Four message types, `add_messages`, `trim_messages` | State for agents *is* a message list |
| 03 | [Tools](#lesson-03-tools) | `@tool`, schemas, `ToolNode`, error defaults | Tools are what makes a graph an agent |
| 04 | [Structured output](#lesson-04-structured-output) | `with_structured_output`, schemas, `Literal` | Typed values to route on |
| 05 | [Model and tools in a graph](#lesson-05-model-and-tools-in-a-graph) | `bind_tools` + `ToolNode` + `tools_condition` | The agent loop, from the parts |
| 06 | [create_agent](#lesson-06-create_agent) | The loop in one call; it is a compiled graph | Know exactly what you are getting |
| 07 | [State, context, memory](#lesson-07-state-context-and-memory) | `AgentState`, `ToolRuntime`, checkpointer, store | Where each kind of data belongs |
| 08 | [Middleware hooks](#lesson-08-middleware-hooks) | Six hooks, node-style vs wrap-style, order | How to customize an agent without rewriting it |
| 09 | [Prebuilt middleware](#lesson-09-prebuilt-middleware) | Limits, retries, fallback, summarization, PII | Production reliability in one line each |
| 10 | [Human-in-the-loop](#lesson-10-human-in-the-loop) | `HumanInTheLoopMiddleware`, approve/edit/reject | Safe tools, built on `interrupt()` |
| 11 | [Streaming](#lesson-11-streaming) | `updates`, `messages`, `custom`, `values`, v2/v3 | UX for slow agents |
| 12 | [Multi-agent and dropping down](#lesson-12-multi-agent-systems-and-dropping-down) | Subagent-as-tool vs agents-as-nodes | When `create_agent` is not enough |
| ★ | [Capstone](#capstone-help-desk-with-approval-gate) | Everything together | A LangGraph workflow built from LangChain agents |

---

## Track A: The building blocks (01-05)

Goal of the track: you can read any LangGraph node's inputs and outputs, define tools and typed outputs, and
assemble the agent loop yourself from LangChain parts.

### Lesson 01: Runnables and chat models

**File:** [`lessons/01_runnables_and_models.py`](lessons/01_runnables_and_models.py) · **Run:** `python lessons/01_runnables_and_models.py` ·
**Prerequisites:** basic Python, type hints.

**Overview.** Everything in LangChain and LangGraph shares one interface, the **Runnable**: `invoke`,
`stream`, `batch` and their async twins. Chat models, prompt templates, output parsers and *compiled
graphs* are all Runnables, and `a | b` chains two of them. This lesson shows that a graph is just another
Runnable and that a node can be any Runnable, not only a function.

**Learning objectives.** By the end of this lesson you will be able to:
1. **Identify** the Runnable interface and **name** which objects implement it: chat model, prompt, parser, chain, compiled graph.
2. **Build** a `prompt | model | parser` chain and wrap a plain function with `RunnableLambda`.
3. **Compose** in both directions: a chain as a graph node, and a graph as a step in a chain.
4. **Apply** a `config` (such as `max_concurrency`) to a Runnable call and **explain** why `batch` needs it with a scripted model.
5. **Choose** between a chain and a graph for a task, with a stated reason.
6. **Swap** the offline model for a real one without changing the surrounding code.

**Concepts covered:** the Runnable interface, chat model, `ChatPromptTemplate`, `StrOutputParser`, LCEL
(`|`), `RunnableLambda`, `batch` and `max_concurrency`, a graph as a Runnable, `get_model` (offline or real).

**Lesson walkthrough.**
1. A `ChatPromptTemplate` turns `{"topic": ...}` into messages.
2. `make_chain` is `prompt | get_model([...]) | StrOutputParser()`: dict in, `str` out.
3. `build_draft_graph` makes that chain (plus a lambda that wraps the text as `{"draft": text}`) a **node** of a one-node `StateGraph`.
4. `graph_as_chain_step` does the reverse: `RunnableLambda | graph | RunnableLambda`.
5. Scripted models are single-use iterators, so every demo builds a fresh one.

**Run it and what you should see.** `invoke -> A graph is state, nodes and edges.`, stream chunks
`['streamed', ' ', 'reply']`, `batch -> ['first', 'second']`, `piped -> HELLO`, the graph's final state
`{'topic': 'graphs', 'draft': 'state plus nodes plus edges'}`, the same draft through the chain step, and a
Mermaid diagram of the one-node graph.

**Common mistakes.** Ending a node-chain with a string (a node must return a state update, usually a dict);
calling `batch` on a scripted model without `max_concurrency=1` (parallel threads race for the scripted
replies); reusing one scripted model for two chains (it runs out of script).

**Exercises (from the file).**
1. Add `config={"run_name": "my_chain", "tags": ["lesson01"]}` to a `chain.invoke` call.
2. Pipe `build_draft_graph()` into a second graph node that title-cases the draft.
3. Put a real key in `.env` and compare `.invoke("hi").usage_metadata` with the scripted model's (`None`).

**Check your understanding.** Each question is tagged with the objective it checks. Try it before opening the answers.
1. [O1] Which of these are Runnables: a prompt template, a chat model, a compiled `StateGraph`, a plain Python function?
2. [O3] In `build_draft_graph` the node is `make_chain(...) | RunnableLambda(lambda text: {"draft": text})`. Why the final lambda?
3. [O4] Why does the lesson call `batch` with `config={"max_concurrency": 1}`?
4. [O5] Task A: summarize, translate, then format an answer. Task B: ask a model for JSON and retry until it parses. Chain or graph for each?
5. [O2] What input does `make_chain("...").invoke(...)` expect, and why a dict?
6. [O6] What would you change to run the lesson against a real Claude model?

<details><summary>Answers</summary>

1. The first three are. A plain function is not, until you wrap it with `RunnableLambda` (`add_node` does that wrapping for you when you pass a function).
2. A graph node must return a **state update**, a dict of state keys. The chain ends in a plain string, so the lambda turns `"text"` into `{"draft": "text"}`.
3. `batch` runs its inputs in parallel threads. A scripted model hands replies out in call order, so parallel calls would race for them and the replies would not line up with the inputs. A real model has no such problem, so you would normally not cap it.
4. A is a **chain**: a straight line with no branching or repetition. B is a **graph**: it needs a cycle (try, check, retry), which is exactly what a chain cannot express.
5. A dict whose keys are the template's variables (`{"topic": ...}`), because the prompt fills `{topic}` from named keys.
6. Nothing in the lesson code. Set `ANTHROPIC_API_KEY` in `.env`; `get_model` then returns `init_chat_model("anthropic:claude-sonnet-5-5")` and the chain, graph and chain step run unchanged.
</details>

### Lesson 02: Messages

**File:** [`lessons/02_messages.py`](lessons/02_messages.py) · **Run:** `python lessons/02_messages.py` · **Prerequisites:** lesson 01.

**Overview.** The state of a chat agent is a list of messages, so reading a message list is reading an agent
trace. This lesson covers the four message types, how a tool call is linked to its result, the
`add_messages` reducer that makes `messages` append (and replace and delete), and how to keep history short.

**Learning objectives.** By the end of this lesson you will be able to:
1. **Name** the four message types and **state** what each is for.
2. **Read** a tool-calling trace and **verify** that each `ToolMessage` links to its request by `tool_call_id`.
3. **Predict** what `add_messages` does for a new id, an existing id, a `RemoveMessage`, and a dict.
4. **Distinguish** `.content`, `.text` and `.content_blocks`, and **use** the right one.
5. **Trim** a long history with `trim_messages` while keeping the system message and a valid start.
6. **Explain** why `MessagesState` and `AgentState` carry the `add_messages` reducer.

**Concepts covered:** `SystemMessage`, `HumanMessage`, `AIMessage` (with `tool_calls`), `ToolMessage`,
`tool_call_id`, message `id`, `RemoveMessage`, `add_messages`, dict-to-message coercion, `trim_messages`.

**Lesson walkthrough.**
1. A five-message conversation: system, human question, AI tool call (`multiply(6, 7)`), tool result `42`, AI answer.
2. `reducer_demo`: `add_messages` with a new id (append), the same id (replace), `RemoveMessage` (delete) and a dict (coerced to `HumanMessage`).
3. `trim_demo`: keep the system message and the tail of a nine-message history.
4. `message_facts`: confirms the link by id and prints `.content_blocks` types.

**Run it and what you should see.** The five-line trace, `linked by id: True`, `content_blocks types: ['text']`,
then four lines for append/replace/remove/coerce, the trimmed history (system message plus the last human and
AI messages), and dict inputs turned into message objects with generated ids.

**Common mistakes.** A `ToolMessage` whose `tool_call_id` does not match the request (providers reject it);
trimming in the middle of a tool call/result pair; assuming `.content` is always a string (it can be a list of blocks).

**Exercises (from the file).**
1. Build a `ToolMessage` with a wrong `tool_call_id` and send the list to a real model.
2. Use `add_messages` to replace the final AI message by reusing its `id`.
3. Change `max_tokens` in `trim_demo` and watch which messages survive; explain `start_on="human"`.

**Check your understanding.**
1. [O2] A `ToolMessage` carries `tool_call_id="x"` but the request used `"call_1"`. What happens and why?
2. [O3] History is `[m1, m2]`. A node returns an `AIMessage` whose id is `m2`. What is the new history?
3. [O3] A node returns `RemoveMessage(id="m1")`. What is the new history?
4. [O5] Why does `trim_demo` pass `start_on="human"`?
5. [O4] When do you use `.text` rather than `.content`?
6. [O6] A node returns `{"messages": [{"role": "user", "content": "hi"}]}`. Why does that work?

<details><summary>Answers</summary>

1. The provider rejects the next request: every tool call in the history must have a matching result, linked by id. The mismatch leaves the call unanswered.
2. Still two messages: `m2` is **replaced in place** by the new message (same id replaces; a new id appends).
3. `[m2]`. `RemoveMessage` is the delete marker the `add_messages` reducer understands.
4. A kept window that starts with an AI or tool message can orphan a tool result (its request was trimmed away) and some providers reject a conversation that does not start with a user turn. `start_on="human"` makes the kept window begin at a human message.
5. When you want the plain string. `.content` can be a string or a list of provider blocks (reasoning, tool calls, images); `.text` is always the concatenated text.
6. `add_messages` (the reducer on the `messages` key) coerces dicts, tuples and strings into message objects and assigns ids.
</details>

### Lesson 03: Tools

**File:** [`lessons/03_tools.py`](lessons/03_tools.py) · **Run:** `python lessons/03_tools.py` · **Prerequisites:** lessons 01-02.

**Overview.** A tool is a Runnable with a name, a description and an argument schema, all generated from a
Python function. The model never runs it; it *asks*, and a `ToolNode` runs the request. This lesson shows
what the model sees, what `ToolNode` does with good calls, bad arguments, unknown tools and tools that raise,
and which of those it turns into messages.

**Learning objectives.** By the end of this lesson you will be able to:
1. **Define** tools with `@tool` and **say** where the name, description and argument schema come from.
2. **Inspect** what the model sees (`name`, `description`, `args`) and **customize** it with `args_schema`.
3. **Run** a tool directly and with a tool-call dict, and **interpret** the returned `ToolMessage`.
4. **Predict** `ToolNode`'s result for success, bad arguments, an unknown tool and a raising tool.
5. **Choose** an error-handling policy for tool exceptions.
6. **Avoid** reserved argument names and **explain** why `ToolNode` must run inside a graph.

**Concepts covered:** `@tool`, docstrings and type hints as the prompt, `args_schema` with Pydantic,
`ToolNode`, `handle_tool_errors`, error `ToolMessage` (`status="error"`), concurrent tool calls.

**Lesson walkthrough.**
1. `multiply` (docstring and hints) and `get_weather` (a Pydantic `args_schema` with a `Literal`).
2. `flaky_lookup` raises for unknown orders.
3. `tool_graph` wraps a `ToolNode` in a one-node graph (it needs graph config), and `run_calls` feeds it an `AIMessage` of requests.
4. The main block prints the model-facing schema, runs tools directly, runs two calls at once, then the four error cases.

**Run it and what you should see.** The tool's name, docstring and `args`; `multiply.invoke -> 42`; a
`ToolMessage` linked to `call_1`; two tool results from one `AIMessage` (`12` and the weather); an
`Error invoking tool 'multiply'...` message for bad arguments; `divide is not a valid tool` for an unknown
tool; `KeyError('order 999 not found')` escaping with the default `ToolNode`; and with
`handle_tool_errors=True` an error `ToolMessage` with `status: error`.

**Common mistakes.** Treating a tool's docstring as documentation only (it is the model's instructions);
assuming `ToolNode` catches everything (a tool's own exception propagates by default); naming a tool
argument `runtime` or `config` (reserved); calling a `ToolNode` outside a graph.

**Exercises (from the file).**
1. Remove the docstring from `multiply`, then the type hints; which is fatal?
2. Add a `Literal["add", "sub"]` argument and call it with a value outside the `Literal`.
3. Give `ToolNode` `handle_tool_errors="Something went wrong..."` (a string) and read the `ToolMessage`.

**Check your understanding.**
1. [O1] Where do a tool's description and its per-argument descriptions come from?
2. [O6] What happens if you call `ToolNode(...).invoke(...)` directly, outside a graph?
3. [O4] One `AIMessage` requests three calls: a good one, one with a string for an `int`, and one to a tool that raises `KeyError`. With a default `ToolNode`, what happens?
4. [O5] You build with `create_agent`, not a hand-made graph. Where do you handle tool exceptions?
5. [O3] An `AIMessage` has two tool calls. How many `ToolMessage`s come back, and how are they run?
6. [O6] Why is `runtime` a bad name for your own tool argument?

<details><summary>Answers</summary>

1. The function's docstring (summary becomes the description; an `Args:` section supplies per-argument descriptions) and its type hints. With `args_schema`, the Pydantic model's docstring and `Field(description=...)` are used instead.
2. `ValueError: Missing required config key ...`. A `ToolNode` needs the graph runtime, so wrap it in a graph (the lesson's `tool_graph`) or use it as a node.
3. The run raises the `KeyError`. Bad arguments and unknown tools are converted into error `ToolMessage`s by default, but a tool's own exception is re-raised. Pass `handle_tool_errors=True` to convert it too.
4. In middleware: `wrap_tool_call` (lesson 08) or `ToolErrorMiddleware` / `ToolRetryMiddleware` (lesson 09). `create_agent` shares the same crash-by-default behavior (lesson 06).
5. Two, one per call, in order. `ToolNode` runs the calls of one message concurrently.
6. `runtime` is the injection point for `ToolRuntime` (state, context, store, `tool_call_id`). It is hidden from the model's schema and filled in by LangChain, so your own argument would collide with it.
</details>

### Lesson 04: Structured output

**File:** [`lessons/04_structured_output.py`](lessons/04_structured_output.py) · **Run:** `python lessons/04_structured_output.py` ·
**Prerequisites:** lessons 01-03.

**Overview.** LangGraph routes on state, and routing wants typed values, not prose. Structured output asks the
model for data matching a schema and returns a validated object. Under the hood it is a forced tool call
named after the schema, which is why the lesson-03 machinery shows up and why the scripted model can fake it.
The classic use: a classifier node followed by a conditional edge.

**Learning objectives.** By the end of this lesson you will be able to:
1. **Write** a Pydantic schema whose docstring and field descriptions guide the model.
2. **Use** `with_structured_output`, with and without `include_raw=True`, and **interpret** the result.
3. **Wire** a classifier node into a conditional edge that routes on a typed field.
4. **Explain** how structured output is implemented with tool calling.
5. **Anticipate** what happens when the model's data violates the schema.
6. **Choose** `Literal` fields to constrain routes.

**Concepts covered:** `with_structured_output`, Pydantic `Field` constraints, `include_raw`
(`raw`, `parsed`, `parsing_error`), the classifier-plus-router pattern, schema as prompt.

**Lesson walkthrough.**
1. `Triage` (category `Literal`, urgency `1..5`, summary) and `triage_message`, which builds the tool call a real model emits.
2. `extract` and `extract_with_raw` use `with_structured_output(Triage)` on the scripted model.
3. `make_triage_graph`: a `classify` node stores a `Triage`, `route` reads `triage.category`, and one of three team nodes runs.

**Run it and what you should see.** The `category` field's JSON schema; a `Triage` object
`category='billing' urgency=4 ...`; `keys: ['parsed', 'parsing_error', 'raw']` with `parsing_error: None`;
then `billing_team (urgency 4)` and `tech_team (urgency 2)` from the two graph runs.

**Common mistakes.** Vague field descriptions (the model guesses); routing on free text instead of a `Literal`;
no plan for validation failure; using a schema the provider cannot express.

**Exercises (from the file).**
1. Return `urgency=9` (outside `le=5`) and catch the validation error; sketch a retry loop.
2. Add `"sales"` to the `Literal`; `route` falls through to `general_team`. Is that what you want?
3. Switch `Triage` to a `TypedDict` and see what `with_structured_output` returns.

**Check your understanding.**
1. [O4] How can a model that only supports tool calling return a typed object?
2. [O2] What does `include_raw=True` return, and when is that useful?
3. [O5] The model returns `urgency=9` for a field with `le=5`. What happens?
4. [O3] Why does `route` read `triage.category` instead of parsing the model's text?
5. [O1] Where do the model's instructions for each field come from?
6. [O6] Why prefer `Literal["billing", "technical", "other"]` to `str` for `category`?

<details><summary>Answers</summary>

1. The schema is turned into a tool; the model is forced to call it; the call's arguments are validated into a schema instance. The result is a tool call parsed for you.
2. A dict with `raw` (the `AIMessage`, with token usage and the tool-call id), `parsed` (the object, or `None`) and `parsing_error`. Use it for usage accounting or to handle a failed parse without an exception.
3. Validation fails and an exception is raised (the test in `tests/test_lessons.py` asserts it). In a graph you would catch it and loop back, or use `include_raw=True` and branch on `parsing_error`.
4. A typed field cannot be mis-parsed; free text needs fragile string matching. The router stays a trivial lookup that is easy to unit test.
5. The class docstring and each `Field(description=...)`. They are part of the prompt, so treat them as prompt engineering.
6. The model can only choose a listed value, validation rejects anything else, and the router's branches stay in sync with the schema.
</details>

### Lesson 05: Model and tools in a graph

**File:** [`lessons/05_model_and_tools_in_a_graph.py`](lessons/05_model_and_tools_in_a_graph.py) ·
**Run:** `python lessons/05_model_and_tools_in_a_graph.py` · **Prerequisites:** lessons 01-04.

**Overview.** The bridge lesson. You assemble the complete agent loop from LangChain parts inside a LangGraph
graph: a model bound to tools, `ToolNode` to run them, and `tools_condition` to loop until the model stops
asking. This is exactly what `create_agent` builds for you in lesson 06.

**Learning objectives.** By the end of this lesson you will be able to:
1. **Assemble** the loop from `bind_tools`, `MessagesState`, `ToolNode` and `tools_condition`.
2. **Explain** what `bind_tools` returns and why you keep the returned object.
3. **Trace** the message sequence for a two-tool question and **name** each message type.
4. **Explain** why the loop ends and **bound** it with `recursion_limit`.
5. **Justify** adding the system message inside the node rather than storing it in state.
6. **Map** each part to its `learn-langgraph` counterpart (hand-written `run_tools` vs `ToolNode`).

**Concepts covered:** `bind_tools`, `MessagesState`, `ToolNode`, `tools_condition`, the ReAct loop,
`GraphRecursionError`, a per-call system prompt.

**Lesson walkthrough.**
1. Two tools, `multiply` and `add`, and a script: call `multiply(6, 7)`, call `add(42, 8)`, answer.
2. `build_agent` binds the tools, defines `call_model` (prepends the system message), and wires `START -> model -> (tools -> model)* -> END`.
3. The seatbelt demo gives the model fifty endless tool calls and a `recursion_limit` of 6.

**Run it and what you should see.** The diagram body (`model`, `tools` with `__tools = multiply, add`,
edges `model -.-> tools`, `tools --> model`); the six-message trace human, AI call, tool `42`, AI call, tool
`50`, final `6 x 7 + 8 = 50.`; and `GraphRecursionError - Recursion limit of 6 reached...`.

**Common mistakes.** Calling the original model instead of the one `bind_tools` returned; storing the system
message in state so it duplicates; no recursion limit on a loop whose exit depends on a model's choice.

**Exercises (from the file).**
1. Add an `llm_calls` counter to state and route to `END` after three calls.
2. Make `add` raise for negatives; compare `ToolNode(TOOLS)` and `ToolNode(TOOLS, handle_tool_errors=True)`.
3. Print `usage_metadata` after swapping in a real model.

**Check your understanding.**
1. [O1] Which edge closes the loop?
2. [O1] What does `tools_condition` return?
3. [O5] Why is the system message added inside `call_model` rather than stored in state?
4. [O4] The model never stops asking for tools. What stops the run, and what does it raise?
5. [O2] `bind_tools` returns a new runnable. What goes wrong if you ignore it?
6. [O6] Which hand-written part of `learn-langgraph` lesson 05 does `ToolNode` replace?

<details><summary>Answers</summary>

1. `tools -> model`. The conditional edge after `model` sends the run to `tools` or `END`; this edge sends it back to the model.
2. The string `"tools"` when the last `AIMessage` has `tool_calls`, otherwise `END`.
3. State is the saved history. A per-call system message is never stored, so it cannot pile up across loop iterations or checkpoints, and you can change it without rewriting history.
4. The recursion limit. `GraphRecursionError: Recursion limit of N reached without hitting a stop condition`. Each model-plus-tools round trip costs two steps.
5. The model you call has no tools bound, so it never emits `tool_calls` and the loop answers without using tools.
6. The `run_tools` node (the `tools_by_name` lookup and loop). `ToolNode` also runs calls concurrently and converts bad arguments to error messages.
</details>

---

## Track B: create_agent is a LangGraph graph (06-09)

Goal of the track: you know exactly what `create_agent` builds, where each kind of data lives, and how
middleware plugs into the graph.

### Lesson 06: create_agent

**File:** [`lessons/06_create_agent.py`](lessons/06_create_agent.py) · **Run:** `python lessons/06_create_agent.py` ·
**Prerequisites:** lessons 01-05.

**Overview.** `langchain.agents.create_agent` builds the loop you wired by hand in lesson 05 and returns a
`CompiledStateGraph`. It is not a different engine: checkpointers, streaming, interrupts and subgraphs all
work because it *is* a LangGraph graph. This lesson proves that, then shows what it adds and a default you
must know.

**Learning objectives.** By the end of this lesson you will be able to:
1. **Build** an agent with `create_agent`, passing a model object or a `"provider:model"` string.
2. **Verify** that it is a compiled graph with nodes `model` and `tools`.
3. **Show** that it behaves like the hand-built lesson-05 graph.
4. **Return** a typed answer with `response_format` and **read** `structured_response`.
5. **Diagnose** the crash-by-default behavior for a raising tool and the recursion-limit seatbelt.
6. **Recognize** `create_react_agent` in older tutorials as deprecated.

**Concepts covered:** `create_agent` parameters, `CompiledStateGraph`, `AgentState`, `system_prompt`,
`response_format`, input forms (a bare string is a user message), default tool-error behavior, `recursion_limit`.

**Lesson walkthrough.**
1. `build_agent` reuses lesson 05's tools and script, adds a `system_prompt`, and passes extra keyword arguments through.
2. `same_shape_as_lesson_05` compares the two message traces.
3. `structured_agent` adds `response_format=Answer`; the model "calls" a tool named `Answer` to deliver it.
4. `explode` is a tool that raises, run through `create_agent` to show the default.

**Run it and what you should see.** `type: CompiledStateGraph`, nodes `['__start__', 'model', 'tools',
'__end__']`, input keys `['messages']`, output keys `['messages', 'structured_response']`,
`identical trace: True`, the six-message trace from a bare-string input, `structured_response:
value=42 explanation='6 times 7'`, `ValueError escaped: boom`, and `GraphRecursionError`.

**Common mistakes.** Expecting `structured_response` without `response_format`; assuming a tool exception is
shown to the model (it crashes the run until you add middleware); copying `create_react_agent` from an old
tutorial; reusing one scripted model for two agents.

**Exercises (from the file).**
1. Pass `name="calculator"` and print `agent.name`.
2. Use `model="anthropic:claude-haiku-5-5"` with a key; does the trace keep its shape?
3. Add a third tool and compare the diagram before and after.

**Check your understanding.**
1. [O2] Which nodes does the graph of a middleware-free `create_agent` contain?
2. [O4] When is `result["structured_response"]` present, and how does a real model produce it?
3. [O5] A tool raises `ValueError`. What does `agent.invoke` do?
4. [O3] How does the lesson prove `create_agent` matches the hand-built graph?
5. [O6] A blog post imports `create_react_agent` from `langgraph.prebuilt`. What do you do?
6. [O1] What two kinds of value can `model=` take?

<details><summary>Answers</summary>

1. `__start__`, `model`, `tools`, `__end__`. Middleware adds more nodes (lesson 08).
2. Only when you pass `response_format`. The schema is offered to the model as a tool (or provider-native structured output where supported) and the validated object is stored under `structured_response`.
3. It raises the `ValueError` out of `invoke`; nothing converts it to a message. Fix it with `wrap_tool_call` or `ToolErrorMiddleware` (lessons 08-09).
4. It runs both with the same script and asserts the sequence of message types and texts is identical (`same_shape_as_lesson_05`, also in the test suite).
5. Migrate to `langchain.agents.create_agent`; the old function is deprecated in its favor.
6. A chat-model object (what the offline lessons pass) or a string such as `"anthropic:claude-sonnet-5-5"`.
</details>

### Lesson 07: State, context and memory

**File:** [`lessons/07_state_context_and_memory.py`](lessons/07_state_context_and_memory.py) ·
**Run:** `python lessons/07_state_context_and_memory.py` · **Prerequisites:** lessons 02, 03, 06.

**Overview.** Four different places data can live, and mixing them up is the commonest agent design mistake:
**state** (graph state, written via `Command`), **context** (per-run, read-only), **short-term memory** (the
checkpointer, per thread) and **long-term memory** (the store, across threads). Tools reach all of it through
one injected parameter, `ToolRuntime`.

**Learning objectives.** By the end of this lesson you will be able to:
1. **Choose** state, context, checkpointer or store for a given piece of data, with a reason.
2. **Extend** `AgentState` and **pass** `context_schema`/`context=` to an agent.
3. **Read** state and context in a tool through `ToolRuntime`, and **write** state with `Command`.
4. **Demonstrate** that threads are isolated and the store is shared across threads.
5. **Explain** why identity belongs in context, not in a model-supplied argument.
6. **Diagnose** a `Command` that is missing its `ToolMessage`.

**Concepts covered:** `AgentState`, `state_schema`, `context_schema`, `ToolRuntime[Ctx, State]`
(`state`, `context`, `store`, `tool_call_id`, `stream_writer`), `Command(update=...)`, `InMemorySaver`
and `thread_id`, `InMemoryStore` and namespaces.

**Lesson walkthrough.**
1. **A.** `lookup_user` reads `runtime.context.user_id` and returns a `Command` that writes `user_name` and the `ToolMessage`; `greet` reads `runtime.state`.
2. **B.** `memory_agent` has a checkpointer; two turns on thread `t1`, none on `t2`; `get_state` shows the saved messages.
3. **C.** `remember_color` and `recall_color` use `runtime.store` under `("prefs", user_id)`; the second thread reads what the first wrote.

**Run it and what you should see.** `user_name in state: Ada Lovelace`; `greet tool said: Hello, Ada
Lovelace!`; thread `t1` with four messages and `t2` empty; and `store item: {'value': 'green'}` with
`another user: None`.

**Common mistakes.** Returning `Command` without the `ToolMessage` (a `ValueError`); letting the model pass a
`user_id` argument; expecting the store to be per-thread; forgetting `thread_id` when a checkpointer is set.

**Exercises (from the file).**
1. Make `lookup_user` return a plain string; where does `user_name` go?
2. Drop the `ToolMessage` from the `Command` and read the `ValueError`.
3. Swap `InMemorySaver` for `SqliteSaver` and show memory surviving a restart.

**Check your understanding.**
1. [O1] Where would you keep: the user's id, the items in this conversation's cart, and the user's favorite color across conversations?
2. [O3] Why can the model not see or forge `runtime`?
3. [O6] A tool returns `Command(update={"user_name": "Ada"})` with no `ToolMessage`. What happens?
4. [O4] You invoke on thread `a`, then on thread `b` with the same user. What does `b` see?
5. [O5] Why is `user_id` in context rather than a tool argument?
6. [O2] What is the difference between `state_schema` and `context_schema`?

<details><summary>Answers</summary>

1. User id: **context** (set by the app). Cart: **state** (a key on a custom `AgentState`, saved per thread). Favorite color across conversations: the **store** (long-term, keyed by user).
2. `runtime` is injected by LangChain at call time and removed from the schema sent to the model, so the model neither sees it nor can supply a value for it.
3. `ValueError: Expected to have a matching ToolMessage in Command.update for tool ...`. Returning a `Command` replaces the normal tool result, so you must supply the `ToolMessage` linked by `runtime.tool_call_id`.
4. Nothing from `a`'s message history (threads are isolated), but the store entry `("prefs", user_id)` is visible, because the store is shared across threads.
5. A model-supplied argument can be wrong or steered by prompt injection. Context is set by the application from the authenticated session, so the tool trusts it.
6. `state_schema` adds keys to the graph state: saved by the checkpointer, visible to every node, writable via updates. `context_schema` types read-only per-run data passed with `context=`; it is not saved and tools cannot change it.
</details>

### Lesson 08: Middleware hooks

**File:** [`lessons/08_middleware_hooks.py`](lessons/08_middleware_hooks.py) · **Run:** `python lessons/08_middleware_hooks.py` ·
**Prerequisites:** lessons 06-07.

**Overview.** Middleware customizes `create_agent` without rewriting the graph. The key distinction:
**node-style** hooks (`before_agent`, `before_model`, `after_model`, `after_agent`) become real *nodes* that
update state, while **wrap-style** hooks (`wrap_model_call`, `wrap_tool_call`) wrap a call and decide how to
make it. Seeing hooks as nodes is what makes `create_agent` feel like LangGraph.

**Learning objectives.** By the end of this lesson you will be able to:
1. **Classify** a hook as node-style or wrap-style and **say** what each can change.
2. **Predict** the execution order for `middleware=[A, B]`.
3. **Build** a budget in state enforced with `jump_to` and `can_jump_to`.
4. **Change** what the model sees with `wrap_model_call`/`@dynamic_prompt` without touching state.
5. **Convert** a tool failure into a message with `wrap_tool_call`.
6. **Map** hooks onto LangGraph ideas (nodes, `Command(goto=END)`) and **choose** class vs decorator form.

**Concepts covered:** `AgentMiddleware`, `@before_model`/`@after_model`/`@wrap_model_call`/`@wrap_tool_call`/
`@dynamic_prompt`, custom state via `state_schema`, `jump_to`, `ModelRequest`/`ModelResponse`,
`request.override(...)`, node names `<Middleware>.<hook>`.

**Lesson walkthrough.**
1. **A.** A `Tracer` class implements all six hooks; two instances (`A`, `B`) log into `TRACE`; the graph's node list shows the hooks.
2. **B.** `count_calls` (an `after_model` hook) increments `model_calls`; `enforce_budget` (a `before_model` hook with `can_jump_to=["end"]`) stops the loop with a defined message.
3. **C.** `@dynamic_prompt` sets a system prompt from the message count; a `@wrap_model_call` records what the model was sent.
4. **D.** `@wrap_tool_call` catches the tool exception and returns an error `ToolMessage`; the model then recovers.
5. **Gotcha.** `undeclared_jump_agent` returns `jump_to` without `can_jump_to`, and nothing happens.

**Run it and what you should see.** Graph nodes including `TracerA.before_model`, `TracerA.after_model`, etc.;
the trace `A.before_agent, B.before_agent, A.before_model, B.before_model, A.wrap_model_call:enter,
B.wrap_model_call:enter, B...exit, A...exit, B.after_model, A.after_model` repeating per model call and ending
`B.after_agent, A.after_agent`; `model_calls: 2 | last message: Stopped: model-call budget used up.`;
`jump_to without can_jump_to -> The model answered normally. (the jump was ignored)`; system prompts `Be
brief.` then `You have used 3 messages; wrap up now.`; and the recovered tool-error trace.

**Common mistakes.** Forgetting `can_jump_to` (the `jump_to` is **silently ignored**, with no error); putting
a state key in a hook without a `state_schema`; expecting `after_*` hooks to run first-to-last (they run
last-to-first); passing two instances of one middleware class (`AssertionError: Please remove duplicate
middleware instances`; the lesson's `Tracer` overrides `name` so `A` and `B` can coexist).

**Exercises (from the file).**
1. Swap the order to `[B, A]` and say which trace lines flip.
2. Delete `can_jump_to=["end"]` from `enforce_budget` and rerun; explain why a silent no-op is worse than an error.
3. Write a `wrap_model_call` that retries a failing call three times.

**Check your understanding.**
1. [O1] Which hook styles can update state, and which can retry a model call?
2. [O2] With `middleware=[A, B]`, in what order do `before_model`, `wrap_model_call` and `after_model` run?
3. [O3] How does the lesson enforce a model-call budget, where is the count stored, and what goes wrong if you forget `can_jump_to`?
4. [O5] You want a tool failure to reach the model as text. Which hook, and what does it return?
5. [O6] Which LangGraph construct is a node-style hook, and which is `{"jump_to": "end"}`?
6. [O6] When should you write class-based middleware instead of a decorator?

<details><summary>Answers</summary>

1. Node-style hooks update state (return a dict or `None`). Wrap-style hooks receive a `handler` and can call it zero, one or several times, so they can retry or short-circuit a model or tool call.
2. `before_model`: A then B. `wrap_model_call`: nested, A outermost (A enter, B enter, model, B exit, A exit). `after_model`: B then A.
3. An `after_model` hook increments `model_calls` in state (declared in a `state_schema`); a `before_model` hook returns `{"jump_to": "end"}` plus a final message once the count reaches the limit.
4. `wrap_tool_call`; it catches the exception and returns a `ToolMessage` (with `tool_call_id=request.tool_call["id"]` and `status="error"`).
5. A node-style hook is a node added to the graph (check `get_graph().nodes`). `jump_to: "end"` is the same idea as `Command(goto=END)`.
6. When one middleware needs several hooks, init-time configuration, extra state or tools, or async variants of its hooks.
</details>

### Lesson 09: Prebuilt middleware

**File:** [`lessons/09_prebuilt_middleware.py`](lessons/09_prebuilt_middleware.py) · **Run:** `python lessons/09_prebuilt_middleware.py` ·
**Prerequisites:** lessons 06-08.

**Overview.** The reliability toolbox. `learn-langgraph` lesson 12 builds retries, budgets and error handling by
hand; for `create_agent`, most of that is one middleware class. Each demo triggers the failure it fixes and
shows the fix working offline.

**Learning objectives.** By the end of this lesson you will be able to:
1. **Select** the right prebuilt middleware for a named problem.
2. **Configure** model and tool call limits and **predict** their exit behavior.
3. **Add** retry for flaky tools and models and **order** retries relative to fallbacks.
4. **Convert** tool exceptions to messages with `ToolErrorMiddleware`.
5. **Trim** context with `SummarizationMiddleware` and **protect** data with `PIIMiddleware`.
6. **Compare** `ToolRetryMiddleware` with `ToolErrorMiddleware` and say when to use each.

**Concepts covered:** `ModelCallLimitMiddleware`, `ToolCallLimitMiddleware`, `ToolRetryMiddleware`,
`ToolErrorMiddleware`, `ModelRetryMiddleware`, `ModelFallbackMiddleware`, `SummarizationMiddleware`,
`PIIMiddleware`, middleware order.

**Lesson walkthrough.**
1. **Limits:** a model that always wants a tool is cut off at three calls; a second `add` call is blocked after a limit of one.
2. **Tool failures:** `flaky_api` fails twice then works (retried); `always_fails` becomes an error message.
3. **Model failures:** a model that fails once is retried; a model that is down falls back to a backup.
4. **Retry then fall back:** the same down provider under three orderings, to show that order and `on_failure` decide whether the fallback ever fires.
5. **Context and privacy:** an email is redacted before the model sees it; a long conversation is summarized.

**Run it and what you should see.** `ModelCallLimit(run_limit=3): 3 real model calls, then -> Model call limits
exceeded: run limit (3/3)`; the second `add` replaced by `Tool call limit exceeded. Do not call 'add' again.`;
`ToolRetry: tool ran 3 times`; `ToolError: ["ValueError: cannot handle 'x'"]`; `ModelRetry: model attempted 2
times`; `ModelFallback -> Answered by the backup model.`; three retry-then-fallback lines (primary called `3x` then the
backup answers; `3x` then `Model call failed after 3 attempts...`; `1x` then the backup answers); `PII redacted: My email is [REDACTED_EMAIL], ...`; and
a history that starts with a summary message followed by the most recent messages.

**Common mistakes.** Using retries for errors that will never succeed (a `ValueError` from bad input);
expecting a fallback to fire after a retry that has the default `on_failure="continue"` (it turns the final
failure into an AI message, so no exception reaches the fallback); assuming limits raise (the default `exit_behavior`
ends the run or blocks the call with a message); expecting `PIIMiddleware` to scan outputs or tool results (off
by default).

**Exercises (from the file).**
1. Explain the three retry-then-fallback lines: why is the primary called 3, 3 and 1 times, and why does only the middle one print a failure?
2. Use `exit_behavior="error"` on `ToolCallLimitMiddleware` and catch the exception.
3. Write a `PIIMiddleware` with a custom regex detector for an order number.

**Check your understanding.**
1. [O1] Name the middleware for: a provider outage, a flaky network tool, a runaway agent loop.
2. [O2] What does `ToolCallLimitMiddleware(tool_name="add", run_limit=1)` do to the second `add` call by default?
3. [O3] You want "retry the primary twice, then switch to a backup". How do you order and configure the two middleware?
4. [O6] `ToolRetryMiddleware` vs `ToolErrorMiddleware`: when each?
5. [O5] What does `SummarizationMiddleware` leave in the history?
6. [O5] Which parts of a conversation does `PIIMiddleware` check by default?

<details><summary>Answers</summary>

1. `ModelFallbackMiddleware`; `ToolRetryMiddleware`; `ModelCallLimitMiddleware`.
2. It blocks the call: the model receives a `ToolMessage` saying the limit was exceeded and not to call it again, and the run continues (the default `exit_behavior="continue"`).
3. List the fallback first (outermost) and the retry after it (innermost), and give the retry `on_failure="error"`. The retry then re-raises after its last attempt and the outer fallback catches it (primary called 3 times, then the backup answers). With the default `on_failure="continue"` the retry swallows the failure into an AI message and the fallback never fires; with the retry listed first (outer), the fallback catches the first failure and the primary is called only once.
4. `ToolRetryMiddleware` for **transient** faults that may succeed on a second try (timeouts, rate limits). `ToolErrorMiddleware` for **permanent** failures, so the model reads the error and changes course. They combine: retry first, then report.
5. A summary message standing in for the older history, followed by the most recent messages (`keep`). The trigger (messages or tokens) decides when it runs.
6. User input (`apply_to_input=True`). Model output and tool results are opt-in (`apply_to_output`, `apply_to_tool_results`).
</details>

---

## Track C: Interactive agents and composition (10-12)

Goal of the track: agents that pause for people, stream to a UI, and combine with other agents or a custom graph.

### Lesson 10: Human-in-the-loop

**File:** [`lessons/10_human_in_the_loop.py`](lessons/10_human_in_the_loop.py) · **Run:** `python lessons/10_human_in_the_loop.py` ·
**Prerequisites:** lessons 07-09 (and `learn-langgraph` lesson 08 helps).

**Overview.** `HumanInTheLoopMiddleware` is `interrupt()` and `Command(resume=...)` pre-built for the most
common case: a person must approve a tool call before it runs. It is an `after_model` node that pauses
between the model's proposal and the tools node, so no side effect can run first. Approve, edit or reject.

**Learning objectives.** By the end of this lesson you will be able to:
1. **Configure** which tools need approval and which decisions are allowed.
2. **Read** the interrupt payload (`action_requests`, `review_configs`).
3. **Resume** with approve, edit and reject decisions and **predict** the effect of each.
4. **Prove** that nothing happens before the decision.
5. **Explain** why a checkpointer and a stable `thread_id` are required.
6. **Gate** only risky calls with a `when` predicate.

**Concepts covered:** `HumanInTheLoopMiddleware(interrupt_on=...)`, `allowed_decisions`, `when`,
`result["__interrupt__"]`, `Command(resume={"decisions": [...]})`, the middleware as a graph node.

**Lesson walkthrough.**
1. `lookup_order` is read-only (`False`); `issue_refund` allows approve, edit, reject.
2. `start` runs to the pause and returns the invoke result containing `__interrupt__`.
3. `decide` resumes with one decision.
4. Four scenarios (approve, edit, reject, conditional approval) check the refund store before and after.

**Run it and what you should see.** Nodes including `HumanInTheLoopMiddleware.after_model`; the paused
request `issue_refund {'order_id': '123', 'amount': 79.0}`; the graph waiting at that node with `REFUNDS: []`;
approve gives `REFUNDS: [('123', 79.0)]`; edit gives `[('123', 40.0)]`; reject gives `[]` and a tool message
`User rejected the tool call for issue_refund with reason: ...`; the `when` demo runs a $20 refund without asking
and pauses a $90 one.

**Common mistakes.** No checkpointer (a pause cannot exist without a saved checkpoint); a different `thread_id`
on resume; resuming with the wrong number of decisions; using `respond` where `reject` was meant for a
side-effecting tool.

**Exercises (from the file).**
1. Remove `checkpointer=` and read the error.
2. Resume with a `thread_id` that was never paused.
3. Add a rule requiring approval for order ids starting with `VIP`; write it as `when` and as a custom hook.

**Check your understanding.**
1. [O4] When the graph is paused, what is `REFUNDS`, and what does `get_state(...).next` show?
2. [O3] The human edits the amount from 79 to 40. What runs?
3. [O3] The human rejects. What does the model see, and what does the tool do?
4. [O5] Why is a checkpointer required?
5. [O6] What does the `when` predicate receive and return?
6. [O2] How many decisions does a batched resume need?

<details><summary>Answers</summary>

1. Empty. The graph is waiting at `('HumanInTheLoopMiddleware.after_model',)`, which runs before the tools node.
2. `issue_refund` runs with the edited arguments (`amount=40.0`); the original proposal is discarded.
3. The tool never runs. The model gets a `ToolMessage` saying the user rejected the call, with the reason you supplied, so it can adapt.
4. A pause is a saved checkpoint. Without one there is nothing to resume from. Use a durable checkpointer (SQLite, Postgres) in production so a pause survives a restart.
5. It receives the proposed tool-call request and returns `True` when approval is required (here `args["amount"] > 50`).
6. One per gated action request in the interrupt, in the same order (`{"decisions": [...]}`).
</details>

### Lesson 11: Streaming

**File:** [`lessons/11_streaming.py`](lessons/11_streaming.py) · **Run:** `python lessons/11_streaming.py` ·
**Prerequisites:** lessons 06-07.

**Overview.** Agents are slow, so products stream. Because `create_agent` returns a graph, you use LangGraph's
`stream_mode`s; this lesson applies them to an agent and adds LangChain's tool-side `stream_writer` and the
newer v3 event streaming.

**Learning objectives.** By the end of this lesson you will be able to:
1. **Choose** `updates`, `messages`, `custom` or `values` for a UI need.
2. **Stream** with `version="v2"` and **dispatch** on `chunk["type"]`.
3. **Emit** custom progress events from a tool with `runtime.stream_writer`.
4. **Rebuild** a streamed answer from `messages` chunks filtered by `langgraph_node`.
5. **Combine** several modes in one stream.
6. **Describe** what `stream_events(version="v3")` adds and why to treat it as experimental.

**Concepts covered:** `stream_mode` (`updates`, `messages`, `custom`, `values`), `version="v2"` chunks
(`type`, `ns`, `data`), `ToolRuntime.stream_writer`, `stream_events` v3 and `.interleave`, an offline model
that streams.

**Lesson walkthrough.**
1. `get_weather` writes `looking up <city>...` to the custom stream, then returns.
2. `updates`, `tokens`, `custom_events`, `mixed`, `values_sizes` each stream the same agent in one mode.
3. `event_stream` uses `stream_events(..., version="v3")` and `.interleave("messages", "tool_calls")`.

**Run it and what you should see.** Updates `model`, `tools`, `model` with their message types; tokens
rebuilt as `It is sunny in Paris.`; custom `['looking up Paris...']`; mixed chunk types `['updates', 'custom',
'updates', 'updates']`; value sizes `[1, 2, 3, 4]`; and a v3 stream of `tool call: get_weather({'city': 'Paris'})`
then `message: It is sunny in Paris.`.

**Common mistakes.** Not filtering `messages` chunks by node (tool-node tokens mix in); calling a tool that uses
`stream_writer` outside a graph run; relying on the v3 API as if it were stable (it emits a
`LangChainBetaWarning`); forgetting `subgraphs=True` when an agent is a subgraph.

**Exercises (from the file).**
1. Use `updates` without `version="v2"` and compare the chunk shape.
2. Stream `["messages", "custom"]` and print each type in order.
3. Re-run `tokens()` with a real key and compare token granularity.

**Check your understanding.**
1. [O1] Which mode for a typing effect, a progress bar of agent steps, and a tool's live status?
2. [O2] What does a v2 chunk look like, and what does it make easier?
3. [O4] Why filter `messages` chunks on `metadata["langgraph_node"] == "model"`?
4. [O3] Why does `runtime.stream_writer` fail when you call the tool directly?
5. [O5] How do you get updates and custom events in one loop?
6. [O6] Why avoid hard-coding v3 in a long-lived product today?

<details><summary>Answers</summary>

1. `messages`; `updates`; `custom`.
2. A dict `{"type": mode, "ns": ..., "data": ...}`. With several modes you can branch on `chunk["type"]` instead of unpacking `(mode, data)` tuples.
3. The stream includes tokens from every LLM call in the graph (and other nodes' metadata). Filtering keeps just the answer you are displaying.
4. The writer belongs to a running graph; outside one there is nowhere to send the event.
5. Pass a list: `stream_mode=["updates", "custom"]`; each chunk says which mode it came from.
6. In `langgraph 1.2.21` it is flagged experimental, so its API may change. Isolate it behind one function (as the lesson does).
</details>

### Lesson 12: Multi-agent systems and dropping down

**File:** [`lessons/12_multi_agent_and_dropping_down.py`](lessons/12_multi_agent_and_dropping_down.py) ·
**Run:** `python lessons/12_multi_agent_and_dropping_down.py` · **Prerequisites:** lessons 04, 06, 07.

**Overview.** Two ways to combine agents, both following from "an agent is a graph": wrap a **specialist as a
tool** so a supervisor keeps control, or drop down to a `StateGraph` and use **agents as nodes** so the graph
decides the flow. This lesson builds both and gives you a rule for choosing, which is the honest answer to
"when would you not use `create_agent` alone?"

**Learning objectives.** By the end of this lesson you will be able to:
1. **Build** a supervisor that delegates to a subagent through a tool.
2. **Explain** what crosses the supervisor/subagent boundary and what stays hidden.
3. **Embed** compiled agents as nodes in a `StateGraph` that shares the `messages` key.
4. **Route** with a structured-output classifier and a conditional edge.
5. **Inspect** the nested structure with `get_graph(xray=True)`.
6. **Choose** between a single agent, subagents-as-tools, and agents-as-nodes, using stated criteria.

**Concepts covered:** subagent as a tool, supervisor, agents as subgraph nodes, `name=`, shared state keys,
the router pattern, `xray`.

**Lesson walkthrough.**
1. **Pattern 1.** `researcher` is a full agent with a tool; `call_researcher` is a `@tool` that calls it and returns only the final text; `supervisor` uses that tool.
2. **Pattern 2.** `specialist(name, reply)` builds a named agent; `build_router_graph` adds `triage` (structured output), a conditional edge, and the two agents as nodes.
3. The final block prints the collapsed and expanded node lists.

**Run it and what you should see.** The supervisor's four-message history (the researcher's own tool loop is
hidden); `team: billing` with the human message and the billing reply and `no duplicated user message: True`;
`team: technical -> Tech here: ...`; and the node lists `['__start__', 'triage', 'billing', 'technical',
'__end__']` (collapsed) versus `[..., 'billing:model', 'technical:model']` (xray).

**Common mistakes.** Returning the subagent's whole message list to the supervisor (defeats the clean
context); forgetting to give agents a `name`; passing state the subagent's schema does not declare; adding a
checkpointer to both the outer graph and the inner agent (let the outer graph own persistence).

**Exercises (from the file).**
1. Pass a state key into and out of the researcher with `ToolRuntime` and `Command`.
2. Add a third specialist and count the places to edit.
3. Add a human approval node between `triage` and the specialists.

**Check your understanding.**
1. [O2] What does the supervisor's message history contain after delegating, and what does it not?
2. [O3] Why does the router graph not duplicate the user's message when an agent runs as a node?
3. [O6] A new requirement: a human approves before any specialist runs. Which pattern, and why?
4. [O6] One domain with six tools and no delegation. Which pattern?
5. [O5] What does `xray=True` change in `get_graph()`?
6. [O4] Where does the routing decision live in Pattern 2, and how is it typed?

<details><summary>Answers</summary>

1. It contains the human message, the supervisor's tool call, one `ToolMessage` holding the subagent's final text, and the final answer. It does not contain the subagent's own tool calls and intermediate messages.
2. The parent and the agent share the `messages` key and its `add_messages` reducer; the agent returns the history including the original human message (same id), which the reducer replaces rather than appends.
3. Agents as nodes (Pattern 2): add an approval node and an edge in the graph. With subagents-as-tools the call happens inside a tool, so inserting a human step between routing and the specialist is awkward.
4. A single `create_agent`. Do not add structure you do not need.
5. It expands subgraphs into the diagram, so you see nodes such as `billing:model` instead of one opaque `billing` node.
6. In the `triage` node, which returns a `Route` schema with a `team: Literal[...]` field; `pick` reads it for the conditional edge.
</details>

---

## Capstone: Help desk with approval gate

**File:** [`capstone/helpdesk_agent.py`](capstone/helpdesk_agent.py) · **Run:** `python capstone/helpdesk_agent.py` ·
**Prerequisites:** lessons 04, 07-10, 12.

**Overview.** One program that uses the whole course: an outer `StateGraph` you control (a structured-output
triage node, a conditional edge, a plain FAQ node) with a `create_agent` as the refund desk. The agent has
identity in context, a model-call budget, tool errors as messages, and human approval before any refund.

**Learning objectives.** By the end of this project you will be able to:
1. **Combine** structured routing, a custom graph, an embedded agent, middleware and persistence in one app.
2. **Place** identity in context and **enforce** ownership checks inside the tools.
3. **Gate** a sensitive tool with `HumanInTheLoopMiddleware` and **pause and resume across a subgraph boundary**.
4. **Bound** the run with `ModelCallLimitMiddleware` and report tool failures as messages.
5. **Prove** with tests that nothing is refunded before approval, a rejection refunds nothing, and users cannot see each other's orders.
6. **Present** the project as a portfolio piece with a diagram, tests and a clear failure story.

**Concepts covered:** agent as a subgraph node, one checkpointer owned by the outer graph, `context_schema`
on both graph and agent, `ToolRuntime`, `ToolErrorMiddleware`, `ModelCallLimitMiddleware`,
`HumanInTheLoopMiddleware`, `Command(resume=...)`.

**Lesson walkthrough.**
1. Tools: `lookup_order` and `issue_refund` both check `ORDERS[id]["owner"] == runtime.context.user_id`.
2. Agent: named `refund_desk`, with the three middleware, `context_schema=Ctx`, and no checkpointer of its own.
3. Outer graph: `triage` (structured `Triage`), `pick` (conditional edge), `refund_desk` (the agent), `faq`; compiled with `InMemorySaver`.
4. `run_refund` drives the flow: ask, pause, then resume with a decision, recording the refund store at each point.

**Run it and what you should see.** `PAUSED for human: issue_refund {'order_id': '123', 'amount': 79.0}`,
`refunds before approval: []`, the six-message trace ending `Done - your refund has been processed.`, and
`refunds after approval: [{'order_id': '123', 'amount': 79.0}]`; on rejection `refunds after rejection: []`;
asking about another user's order gives `['Tool failed: order 456 not found']`; and a non-refund message
goes straight to the FAQ answer.

**Common mistakes.** Giving the inner agent its own checkpointer (the outer graph should own it); trusting a
user id from tool arguments; letting a rejection leave a tool call unanswered; no model-call budget.

**Extensions (not in the file).**
1. Swap `InMemorySaver` for SQLite or Postgres and resume after a process restart.
2. Only require approval above a threshold with `when=`.
3. Stream the run with `stream_mode=["updates", "messages"]`, `subgraphs=True`, `version="v2"`.
4. Add an eval set of support conversations with assertions on tool calls and the refund store.

**Check your understanding.**
1. [O2] A user asks about order 456, which belongs to someone else. What does the model learn, and why?
2. [O3] Why does the inner agent have no checkpointer?
3. [O3] The interrupt is raised inside the `refund_desk` subgraph. How does the caller resume it?
4. [O5] Which two assertions make the approval-gate test meaningful?
5. [O4] What happens when the model keeps calling tools forever?
6. [O6] Name one risk you would call out when presenting this project, and the next step you would take.

<details><summary>Answers</summary>

1. It gets a tool error `order 456 not found`, the same response as for a missing order, so it cannot confirm that another customer's order exists. Ownership is checked against `runtime.context.user_id`, which the application set.
2. The outer graph owns persistence; the subgraph inherits it. A second checkpointer would split the saved state.
3. Call the outer graph again with `Command(resume={"decisions": [...]})` and the same `thread_id` and `context`. The pause and resume cross the subgraph boundary.
4. The refund store is empty after the pause (nothing happened before approval), and after the decision it contains exactly the intended refund (approve) or nothing (reject).
5. `ModelCallLimitMiddleware(run_limit=...)` ends the run with a defined message once the budget is reached; the recursion limit remains the seatbelt.
6. For example: state is in memory, so a restart loses pending approvals (next step: a durable checkpointer); or there is no approval threshold or expiry (next step: add `when=` and a timeout, and an eval set).
</details>

---

## Study plans

**One week, focused**
- **Days 1-2:** lessons 01-03. Do *every* exercise. Re-type lesson 03's `tool_graph` from memory.
- **Day 3:** lessons 04-05. Draw the lesson-05 graph on paper, then compare with `get_graph().draw_mermaid()`.
- **Day 4:** lessons 06-07. Put a key in `.env` and re-run both with a real model; watch the tool calls.
- **Days 5-6:** lessons 08-10. Write one custom hook and one `when` rule of your own.
- **Day 7:** lessons 11-12 and the capstone.

**Alongside `learn-langgraph`:** take 01-04 here, then `learn-langgraph` 01-04 and 05-06 (which will now read
as review), then 06-10 here with `learn-langgraph` 07-10, then 11-12 with its 11.

**If you already use `create_agent`:** 05, 08, 12 (what is really in the graph, how hooks run, when to drop down).

## Concepts in one page

- **A graph is a Runnable.** `invoke`, `stream`, `batch`; a node can be a chain; a graph can be a chain step.
- **State for agents is a message list.** Four types; `add_messages` appends, replaces by id, deletes with `RemoveMessage`.
- **A tool is a name, a description and an argument schema.** Docstrings and hints are the model's instructions. `ToolNode` runs the calls (concurrently) and, by default, lets a tool's own exceptions crash the run.
- **Structured output is a forced tool call.** Use `Literal` fields to give a conditional edge something safe to route on.
- **`create_agent` returns a compiled LangGraph graph** with nodes `model` and `tools`. Everything LangGraph offers works on it.
- **Four homes for data:** state (thread, writable via `Command`), context (one run, read-only), checkpointer (short-term memory), store (long-term, cross-thread). Identity goes in context.
- **Middleware hooks are nodes.** Node-style hooks update state; wrap-style hooks wrap a call. `before_*` first-to-last, `wrap_*` nested, `after_*` last-to-first.
- **Budgets beat limits.** Put loop budgets in state or in `ModelCallLimitMiddleware`; set `recursion_limit` anyway.
- **Human approval is `interrupt()`.** `HumanInTheLoopMiddleware` pauses after the model proposes a call and before tools run; it needs a checkpointer and a stable `thread_id`.
- **Stream with the graph's modes.** `updates`, `messages`, `custom`, `values`; `version="v2"` chunks carry a `type`.
- **Compose agents two ways:** as tools (clean context, supervisor in charge) or as nodes (you control the flow). Drop to `StateGraph` when you need explicit routing, approvals or non-message state.
- **Scripted models are single-use.** Build a fresh one per agent, and cap `batch` concurrency when you use one.

## Project layout

```
lessons/        01-12 scripts; common.py (offline/real model switch, scripted fake, show/banner helpers)
capstone/       helpdesk_agent.py
tests/          56 offline tests (+ 1 opt-in live test)
pyproject.toml  Hatch/Hatchling build; pinned dependencies; ruff and ty (strictest) configuration
.env.example    optional API key and model selection
```

`lessons/common.py` holds the offline double: `ScriptedChatModel` replays a list of `AIMessage`s, accepts
`bind_tools` as a no-op, and streams words and tool calls, so every lesson (including streaming) runs
without a key. Its results demonstrate the *wiring*, not model quality: re-run with a real model to see
model behavior.

## Testing and quality gates

```powershell
pytest -q                          # 56 tests, offline
pytest -q -m live                  # opt-in: one test against a real model (needs ANTHROPIC_API_KEY)
ruff check .                       # lint
ty check                           # type check: every rule is an error (see pyproject.toml)
pytest --cov                       # coverage gate: 85% (currently about 97%)
```

| Test file | Tests | Covers |
|---|---|---|
| `tests/test_lessons.py` | 37 | one group per lesson, plus the capstone (approve, reject, ownership, budget) |
| `tests/test_smoke.py` | 14 | runs every lesson script and the capstone exactly as a learner would |
| `tests/test_model_config.py` | 5 | the offline/online switch in `common.py` |
| `tests/test_live.py` | 1 (opt-in) | the same loop against a real model |

## What was left out, and why

This tutorial is deliberately narrow. Not covered, and where to go:

- **LCEL chains, retrievers, document loaders, vector stores, text splitters.** Useful, but not needed to
  understand LangGraph. RAG lives in `learn-langgraph` lessons 13-15.
- **MCP tools, the frontend and generative-UI patterns, Agent Chat UI, LangSmith Studio and deployment.**
  Real topics; see the docs map below.
- **Handoffs and skills (other multi-agent patterns).** Lesson 12 covers the two you need to reason about
  the rest; the docs describe handoffs, router and skills in detail.
- **Provider-specific middleware** (prompt caching, bash and file tools) and the **Deep Agents** harness.
- **Evals and observability.** Set `LANGSMITH_TRACING=true` with a key and every lesson is traced; the docs
  cover evals and unit and integration testing of agents.

## Official docs map

- LangChain overview: <https://docs.langchain.com/oss/python/langchain/overview.md>
- Agents: <https://docs.langchain.com/oss/python/langchain/agents.md>
- Models: <https://docs.langchain.com/oss/python/langchain/models.md>
- Messages: <https://docs.langchain.com/oss/python/langchain/messages.md>
- Tools: <https://docs.langchain.com/oss/python/langchain/tools.md>
- Structured output: <https://docs.langchain.com/oss/python/langchain/structured-output.md>
- Short-term memory: <https://docs.langchain.com/oss/python/langchain/short-term-memory.md>
- Long-term memory: <https://docs.langchain.com/oss/python/langchain/long-term-memory.md>
- Runtime: <https://docs.langchain.com/oss/python/langchain/runtime.md>
- Middleware overview: <https://docs.langchain.com/oss/python/langchain/middleware/overview.md>
- Prebuilt middleware: <https://docs.langchain.com/oss/python/langchain/middleware/built-in.md>
- Custom middleware: <https://docs.langchain.com/oss/python/langchain/middleware/custom.md>
- Human-in-the-loop: <https://docs.langchain.com/oss/python/langchain/human-in-the-loop.md>
- Streaming: <https://docs.langchain.com/oss/python/langchain/streaming.md>
- Event streaming: <https://docs.langchain.com/oss/python/langchain/event-streaming.md>
- Multi-agent (subagents, handoffs, router, skills): <https://docs.langchain.com/oss/python/langchain/multi-agent/index.md>
- Context engineering: <https://docs.langchain.com/oss/python/langchain/context-engineering.md>
- Guardrails: <https://docs.langchain.com/oss/python/langchain/guardrails.md>
- Unit testing agents: <https://docs.langchain.com/oss/python/langchain/test/unit-testing.md>
- LangGraph overview (the engine underneath): <https://docs.langchain.com/oss/python/langgraph/overview.md>
- Full docs index: <https://docs.langchain.com/llms.txt>

> API note: LangChain and LangGraph move fast. If something errors after an upgrade, check the changelog first,
> then run `pytest tests/test_lessons.py`: it pins every behavior this tutorial claims. Examples here were run
> against the versions listed at the top.
