#!/usr/bin/env python3
"""
gaia_gt_pilot.py
================
Instrumented pilot harness for applying Generalizability Theory (GT) to an
API-based multi-agent system on the GAIA Level 1 validation set.

What it does
------------
1. Loads GAIA tasks from a local JSON file (your gaia_level1.json).
2. For each task, runs an open_deep_research-style two-tier smolagents agent
   R times (replications) -- replications are what let a G-study separate
   true-score variance from residual/error variance.
3. Captures a TYPED trajectory for every run (planning steps, tool calls,
   observations, errors, durations) so you can apply HORIZON's failure
   taxonomy downstream.
4. Scores each run with the OFFICIAL GAIA scorer (number/list/string
   normalisation, exact match).
5. Writes ONE JSONL record per (task x replication) with every facet field
   your GT decomposition needs:
       - model            -> the "model" facet
       - prompt_variant   -> manager scaffold facet (tool/code), crossed
       - effort           -> reasoning_effort facet (low/high), crossed
       - temperature      -> held constant at 1 (reasoning models lock temp!=1)
       - task_type        -> derived task-type / domain facet (coarse)
       - level            -> GAIA difficulty level
       - horizon_steps    -> GAIA annotator step count (your "horizon" facet)
       - horizon_tools    -> GAIA annotator tool count (auxiliary)
       - run_index        -> replication index (for the residual term)
       - success          -> 0/1 GAIA-scored outcome (the score)
       - trajectory       -> list of typed steps for HORIZON classification

Setup
-----
    py -3.11 -m venv .venv
    .venv\\Scripts\\Activate.ps1
    pip install -r harness/requirements.txt
    python harness/fetch_gaia.py            # gated HF dataset -> data/private/gaia_level1.json
    $env:OPENAI_API_KEY = "sk-..."          # for gpt-* models
    $env:SERPER_API_KEY = "..."             # default search backend (serper.dev)

Run (from the repo root)
------------------------
    # smoke test: 3 tasks x 2 efforts x 2 scaffolds x 1 run
    python harness/gaia_gt_pilot.py --tasks data/private/gaia_level1.json --model gpt-5.4-mini --runs 1 --limit 3 --out raw/smoke.jsonl

    # the study design: 53 tasks x 2 x 2 x 20 runs (use harness/run_overnight.ps1)

Known limitations of the 2026-06-26 run (see README "Limitations")
-----------------------------------------------------------------
* The timeout did not abandon runs (fixed post hoc in _run_with_timeout).
* GAIA attachments are never given to the agent. Use --exclude-attachments to
  drop those 11 tasks in future runs.
* Runs executed in nested order (task > effort > scaffold > replication), so
  time is confounded with condition. Use --shuffle for future runs.
* Only the MANAGER's memory is serialized. The managed web agent's internal
  searches are not visible in `trajectory`, n_steps or n_tool_calls.
"""

import argparse
import concurrent.futures
import json
import os
import re
import string
import sys
import time
import traceback
from datetime import datetime, timezone

# Force UTF-8 on stdout/stderr. On Windows the default console codec is cp1252,
# and smolagents' rich console printing of trajectories crashes (UnicodeEncodeError,
# 'charmap' codec) whenever a search result contains a non-Latin1 character (e.g.
# 'ō'). That crash surfaces as a spurious tool/step error and contaminates the
# failure data, so we replace the codec before any agent runs.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import litellm
litellm.modify_params = True


# --------------------------------------------------------------------------- #
#  Windows-compatible per-task timeout                                         #
# --------------------------------------------------------------------------- #
def _run_with_timeout(fn, timeout_s, *args, **kwargs):
    # NOTE (post-hoc fix, applied after the 2026-06-26 run): the original version
    # used `with ThreadPoolExecutor(...) as ex:`. Leaving a `with` block calls
    # ex.shutdown(wait=True), so after a TimeoutError the harness still BLOCKED
    # until the agent finished (observed wall times up to 1072 s against a 240 s
    # "limit"), and the run was then scored 0 even if the agent had produced an
    # answer. We now return control immediately. Python threads cannot be killed,
    # so the abandoned agent may keep running (and spending API calls) in the
    # background, but its result is discarded and wall_time_s is honest.
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = ex.submit(fn, *args, **kwargs)
    try:
        return future.result(timeout=timeout_s)
    except concurrent.futures.TimeoutError:
        raise TimeoutError(
            f"Task exceeded {timeout_s}s wall-clock limit and was abandoned."
        )
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


# --------------------------------------------------------------------------- #
#  Official GAIA scorer                                                        #
# --------------------------------------------------------------------------- #
def _is_float(element) -> bool:
    try:
        float(element)
        return True
    except (ValueError, TypeError):
        return False


def normalize_number_str(number_str: str) -> float:
    for ch in ["$", "%", ","]:
        number_str = number_str.replace(ch, "")
    try:
        return float(number_str)
    except ValueError:
        return float("inf")


def split_string(s: str, char_list=(",", ";")):
    pattern = f"[{''.join(char_list)}]"
    return re.split(pattern, s)


def normalize_str(input_str: str, remove_punct: bool = True) -> str:
    no_spaces = re.sub(r"\s", "", input_str)
    if remove_punct:
        translator = str.maketrans("", "", string.punctuation)
        return no_spaces.lower().translate(translator)
    return no_spaces.lower()


def gaia_question_scorer(model_answer: str, ground_truth: str) -> bool:
    if model_answer is None:
        return False
    model_answer = str(model_answer).strip()
    ground_truth = str(ground_truth).strip()
    if _is_float(ground_truth):
        return normalize_number_str(model_answer) == float(ground_truth)
    if any(c in ground_truth for c in [",", ";"]):
        gt_elems = split_string(ground_truth)
        ma_elems = split_string(model_answer)
        if len(gt_elems) != len(ma_elems):
            return False
        comparisons = []
        for ma_elem, gt_elem in zip(ma_elems, gt_elems):
            if _is_float(gt_elem):
                comparisons.append(normalize_number_str(ma_elem) == float(gt_elem))
            else:
                comparisons.append(
                    normalize_str(ma_elem, remove_punct=False)
                    == normalize_str(gt_elem, remove_punct=False)
                )
        return all(comparisons)
    return normalize_str(model_answer) == normalize_str(ground_truth)


# --------------------------------------------------------------------------- #
#  Tolerant GAIA task loader                                                   #
# --------------------------------------------------------------------------- #
def _get(d: dict, *keys, default=None):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default


def load_tasks(path: str):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        for k in ("data", "tasks", "rows", "questions"):
            if isinstance(raw.get(k), list):
                raw = raw[k]
                break
    if not isinstance(raw, list):
        raise ValueError("Could not find a list of tasks in the JSON file.")
    tasks = []
    for r in raw:
        meta = _get(r, "Annotator Metadata", "annotator_metadata", default={}) or {}
        steps = _get(meta, "Number of steps", "num_steps", default=None)
        tools_n = _get(meta, "Number of tools", "num_tools", default=None)
        tasks.append(
            {
                "task_id": str(_get(r, "task_id", "id", default=f"idx_{len(tasks)}")),
                "question": _get(r, "Question", "question", default=""),
                "true_answer": _get(r, "Final answer", "final_answer", "answer", default=""),
                "level": _get(r, "Level", "level", default=1),
                "file_name": _get(r, "file_name", "file", default=""),
                "horizon_steps": _to_int(steps),
                "horizon_tools": _to_int(tools_n),
                "annotator_tools": _get(meta, "Tools", default=""),
            }
        )
    return tasks


def _to_int(x):
    try:
        return int(str(x).strip())
    except (ValueError, TypeError, AttributeError):
        return None


# --------------------------------------------------------------------------- #
#  Coarse task-type tagging                                                    #
# --------------------------------------------------------------------------- #
def coarse_task_type(task: dict) -> str:
    q = (task.get("question") or "").lower()
    fn = (task.get("file_name") or "").lower()
    if fn.endswith((".xlsx", ".csv", ".tsv")):
        return "tabular_file"
    if fn.endswith((".pdf", ".docx", ".txt")):
        return "document_file"
    if fn.endswith((".png", ".jpg", ".jpeg", ".mp3", ".wav", ".mp4")):
        return "media_file"
    if any(w in q for w in ("youtube", "video", "image", "photo")):
        return "multimodal_web"
    if any(w in q for w in ("calculate", "how many", "average", "sum", "percent")):
        return "quantitative_web"
    return "factual_web"


# --------------------------------------------------------------------------- #
#  Serper search tool                                                          #
# --------------------------------------------------------------------------- #
import requests as _requests


def _make_serper_tool():
    from smolagents import Tool

    class SerperSearchTool(Tool):
        name = "web_search"
        description = (
            "Search the web for current information. "
            "Returns titles, URLs and snippets for the top results."
        )
        inputs = {
            "query": {"type": "string", "description": "The search query to look up."}
        }
        output_type = "string"

        def forward(self, query: str) -> str:
            api_key = os.environ.get("SERPER_API_KEY", "")
            if not api_key:
                return "Error: SERPER_API_KEY environment variable is not set."
            try:
                resp = _requests.post(
                    "https://google.serper.dev/search",
                    headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
                    json={"q": query, "num": 5},
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()
                out = []
                for r in data.get("organic", [])[:5]:
                    out.append(
                        f"{r.get('title','')}\n{r.get('link','')}\n{r.get('snippet','')}"
                    )
                return "\n\n".join(out) if out else "No results found."
            except Exception as exc:
                return f"Search error: {exc}"

    return SerperSearchTool()


def _make_tavily_tool():
    """Tavily search backend (LLM-optimised; reads TAVILY_API_KEY)."""
    from smolagents import Tool

    class TavilySearchTool(Tool):
        name = "web_search"
        description = (
            "Search the web for current information. "
            "Returns titles, URLs and snippets for the top results."
        )
        inputs = {
            "query": {"type": "string", "description": "The search query to look up."}
        }
        output_type = "string"

        def forward(self, query: str) -> str:
            api_key = os.environ.get("TAVILY_API_KEY", "")
            if not api_key:
                return "Error: TAVILY_API_KEY environment variable is not set."
            try:
                from tavily import TavilyClient

                client = TavilyClient(api_key=api_key)
                data = client.search(query=query, max_results=5)
                out = []
                for r in data.get("results", [])[:5]:
                    out.append(
                        f"{r.get('title','')}\n{r.get('url','')}\n{r.get('content','')}"
                    )
                return "\n\n".join(out) if out else "No results found."
            except Exception as exc:
                return f"Search error: {exc}"

    return TavilySearchTool()


def _make_ddg_tool():
    """DuckDuckGo backend (free, no key). Retries to blunt rate-limiting."""
    from smolagents import Tool

    class DDGSearchTool(Tool):
        name = "web_search"
        description = (
            "Search the web for current information. "
            "Returns titles, URLs and snippets for the top results."
        )
        inputs = {
            "query": {"type": "string", "description": "The search query to look up."}
        }
        output_type = "string"

        def forward(self, query: str) -> str:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS  # older package name
            last_exc = None
            for attempt in range(3):
                try:
                    with DDGS() as ddgs:
                        results = list(ddgs.text(query, max_results=5))
                    out = []
                    for r in results[:5]:
                        out.append(
                            f"{r.get('title','')}\n"
                            f"{r.get('href') or r.get('url','')}\n"
                            f"{r.get('body') or r.get('content','')}"
                        )
                    return "\n\n".join(out) if out else "No results found."
                except Exception as exc:
                    last_exc = exc
                    time.sleep(2 * (attempt + 1))  # back off on rate limits
            return f"Search error: {last_exc}"

    return DDGSearchTool()


def _make_search_tool(backend: str):
    """Dispatch to the selected web-search backend (held constant per study)."""
    backend = (backend or "serper").lower()
    if backend == "tavily":
        return _make_tavily_tool()
    if backend in ("ddg", "duckduckgo"):
        return _make_ddg_tool()
    return _make_serper_tool()


# --------------------------------------------------------------------------- #
#  Wikipedia-aware visit tool                                                  #
#  For Wikipedia URLs: uses MediaWiki extracts API -> plain article text,      #
#  no navigation clutter. The full HTML page buries data under thousands of   #
#  chars of menus which our truncation limit cuts off before the content.      #
#  For all other URLs: fetches HTML with a real browser UA -> markdownify.    #
# --------------------------------------------------------------------------- #
def _make_visit_tool():
    from smolagents import Tool
    import json as _json

    class WikiAwareVisitTool(Tool):
        name = "visit_webpage"
        description = (
            "Fetch and read the content of a webpage at a given URL. "
            "Works on Wikipedia articles and general web pages."
        )
        inputs = {
            "url": {"type": "string", "description": "The full URL to visit."}
        }
        output_type = "string"

        _UA = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
        _TRUNC = 8000  # max chars returned to the agent

        def forward(self, url: str) -> str:
            import urllib.request
            import re as _re

            # Wikipedia: use the clean MediaWiki extracts API.
            # This returns plain article text with no navigation markup.
            if "wikipedia.org/wiki/" in url:
                try:
                    title = url.split("/wiki/")[-1].split("#")[0]
                    api = (
                        "https://en.wikipedia.org/w/api.php"
                        f"?action=query&prop=extracts&explaintext=true"
                        f"&exsectionformat=plain&format=json"
                        f"&titles={title}&redirects=1"
                    )
                    req = urllib.request.Request(api, headers={"User-Agent": self._UA})
                    with urllib.request.urlopen(req, timeout=15) as resp:
                        data = _json.loads(resp.read().decode("utf-8"))
                    pages = data.get("query", {}).get("pages", {})
                    for page in pages.values():
                        extract = page.get("extract", "")
                        if extract:
                            return extract[: self._TRUNC]
                    return "Wikipedia API returned no content for that title."
                except Exception as exc:
                    return f"Wikipedia API error: {exc}"

            # Generic URL: fetch HTML and convert to text.
            try:
                req = urllib.request.Request(url, headers={"User-Agent": self._UA})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
            except Exception as exc:
                return f"Error fetching {url}: {exc}"

            try:
                import markdownify
                text = markdownify.markdownify(raw, heading_style="ATX")
            except Exception:
                text = _re.sub(r"<[^>]+>", " ", raw)
                text = _re.sub(r"\s+", " ", text).strip()

            return text[: self._TRUNC]

    return WikiAwareVisitTool()


# --------------------------------------------------------------------------- #
#  Agent construction (THE SWAP SEAM)                                          #
# --------------------------------------------------------------------------- #
def build_agent(model_id: str, prompt_variant: str, effort: str, temperature: float,
                search_backend: str = "serper"):
    """
    Returns a fresh agent for ONE run. Built fresh every run so replications
    are independent (no carried-over memory).

    The crossed design facets are:
      * `effort`         -> the model's `reasoning_effort` ("low" / "high"), set on
                            the shared LiteLLM model below. Passed through to
                            litellm.completion; dropped for non-reasoning models.
      * `prompt_variant` -> the manager SCAFFOLD: "tool" builds a ToolCallingAgent
                            manager (acts via JSON tool calls); "code" builds a
                            CodeAgent manager (writes & executes Python, calling the
                            web agent as a function). This replaced temperature as
                            the second factor because gpt-5.x reasoning models lock
                            temperature to 1 whenever reasoning_effort is low/high.
    `temperature` is therefore held constant (1.0) and only set on the model so the
    value is explicit; it is not manipulated. The manager and the managed web agent
    share one model object, so effort/temperature apply uniformly across the run.
    """
    from smolagents import CodeAgent, ToolCallingAgent, LiteLLMModel

    model_kwargs = {"model_id": model_id, "temperature": temperature}
    if effort:
        model_kwargs["reasoning_effort"] = effort
    model = LiteLLMModel(**model_kwargs)

    web_agent = ToolCallingAgent(
        tools=[_make_search_tool(search_backend), _make_visit_tool()],
        model=model,
        max_steps=10,
        name="web_search_agent",
        description="Searches the web and reads pages to gather facts for the manager.",
    )

    if prompt_variant == "tool":
        manager = ToolCallingAgent(
            tools=[],
            model=model,
            managed_agents=[web_agent],
            max_steps=8,
            name="manager",
            description="Plans, delegates to the web agent, and returns the final answer.",
        )
    else:  # 'code'
        manager = CodeAgent(
            tools=[],
            model=model,
            managed_agents=[web_agent],
            max_steps=8,
            additional_authorized_imports=["pandas", "numpy", "math", "datetime", "json", "re"],
        )
    return manager


GAIA_SYSTEM_HINT = (
    "You are a precise general assistant solving a GAIA task. "
    "Work step by step, use the web agent for any fact you are unsure of, "
    "and finish with a single line of the exact form:\n"
    "FINAL ANSWER: <answer>\n"
    "The answer must be a number OR as few words as possible OR a comma-separated "
    "list. Do not add units unless asked. Do not include articles or explanations "
    "in the final answer line."
)


def extract_final_answer(raw) -> str:
    text = str(raw)
    m = re.search(r"FINAL ANSWER:\s*(.+)", text, flags=re.IGNORECASE)
    return m.group(1).strip() if m else text.strip().splitlines()[-1].strip()


# --------------------------------------------------------------------------- #
#  Trajectory serialisation                                                    #
# --------------------------------------------------------------------------- #
def serialize_trajectory(agent):
    steps_out = []
    memory = getattr(agent, "memory", None)
    raw_steps = getattr(memory, "steps", None) or []
    for i, st in enumerate(raw_steps):
        step_type = type(st).__name__
        tool_calls = []
        for tc in getattr(st, "tool_calls", None) or []:
            tool_calls.append(
                {
                    "name": getattr(tc, "name", None),
                    "arguments": _jsonsafe(getattr(tc, "arguments", None)),
                }
            )
        err = getattr(st, "error", None)
        steps_out.append(
            {
                "index": i,
                "step_type": step_type,
                "model_output": _truncate(getattr(st, "model_output", None)),
                "tool_calls": tool_calls,
                "observations": _truncate(getattr(st, "observations", None)),
                "error": None if err is None else f"{type(err).__name__}: {err}",
                "duration_s": getattr(st, "duration", None),
            }
        )
    return steps_out


def _truncate(x, n=4000):
    if x is None:
        return None
    s = x if isinstance(x, str) else json.dumps(_jsonsafe(x), ensure_ascii=False)
    return s if len(s) <= n else s[:n] + f"...[truncated {len(s) - n} chars]"


def _jsonsafe(x):
    try:
        json.dumps(x)
        return x
    except (TypeError, ValueError):
        return str(x)


# Tool names a CodeAgent uses to execute its Python (the final answer is emitted
# by calling final_answer(...) inside such a block, not via a "final_answer" tool).
_CODE_EXEC_TOOLS = ("python_interpreter", "code_interpreter", "python")


def _is_final_answer_call(tc) -> bool:
    name = tc.get("name")
    if name == "final_answer":
        return True
    if name in _CODE_EXEC_TOOLS:
        args = tc.get("arguments")
        code = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False, default=str)
        return "final_answer" in code
    return False


def error_flags(trajectory, success, answer):
    flags = []
    if not answer:
        flags.append("empty_answer")
    if any(s["error"] for s in trajectory):
        flags.append("tool_or_exec_error")
    # Did the agent ever formally terminate via a final_answer call? If not, it
    # ran out of steps / gave up / terminated malformed -- the real "step cap"
    # signal. The two scaffolds finalize differently and BOTH must be recognised,
    # or this flag becomes a scaffold artifact:
    #   * ToolCallingAgent -> a tool call literally named "final_answer".
    #   * CodeAgent        -> executes `final_answer(...)` inside a code block,
    #                         which surfaces as a "python_interpreter" tool call
    #                         whose arguments contain the final_answer(...) call.
    made_final = any(_is_final_answer_call(tc) for s in trajectory for tc in s["tool_calls"])
    if not made_final:
        flags.append("no_final_answer")
    if not any(tc for s in trajectory for tc in s["tool_calls"]):
        flags.append("no_tool_use")
    return flags


# --------------------------------------------------------------------------- #
#  Human-readable per-task .txt writer                                         #
#  One file per task (53 total); each iteration appended as a labelled block.  #
#  These are for eyeballing trajectories; the JSONL is the structured source   #
#  of truth that the CSV aggregator consumes.                                  #
# --------------------------------------------------------------------------- #
def append_txt_block(txt_dir: str, record: dict):
    os.makedirs(txt_dir, exist_ok=True)
    path = os.path.join(txt_dir, f"{record['task_id']}.txt")
    first = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if first:
            f.write("=" * 78 + "\n")
            f.write(f"TASK {record['task_id']}\n")
            f.write(f"model={record['model']}  prompt_variant={record['prompt_variant']}  "
                    f"task_type={record['task_type']}  level={record['level']}\n")
            f.write(f"horizon_steps={record['horizon_steps']}  "
                    f"horizon_tools={record['horizon_tools']}\n")
            f.write("-" * 78 + "\n")
            f.write("QUESTION:\n")
            f.write((record["question"] or "") + "\n")
            f.write(f"TRUE ANSWER: {record['true_answer']}\n")
            f.write("=" * 78 + "\n")
        f.write("\n")
        f.write("#" * 78 + "\n")
        f.write(f"### ITERATION {record['run_index']}   "
                f"effort={record['effort']}  temperature={record['temperature']}   "
                f"[{record['timestamp_utc']}]\n")
        f.write("#" * 78 + "\n")
        f.write(f"model_answer : {record['model_answer']}\n")
        f.write(f"success      : {record['success']}\n")
        f.write(f"wall_time_s  : {record['wall_time_s']}\n")
        f.write(f"n_steps      : {record['n_steps']}\n")
        f.write(f"n_tool_calls : {record['n_tool_calls']}\n")
        f.write(f"error_flags  : {record['error_flags']}\n")
        f.write(f"crash        : {record['crash']}\n")
        f.write("--- trajectory ---\n")
        for st in record["trajectory"]:
            f.write(f"[step {st['index']}] {st['step_type']}"
                    f"  (duration_s={st['duration_s']})\n")
            if st["tool_calls"]:
                for tc in st["tool_calls"]:
                    f.write(f"    TOOL CALL: {tc['name']}  args={tc['arguments']}\n")
            if st["model_output"]:
                f.write(f"    MODEL_OUTPUT: {st['model_output']}\n")
            if st["observations"]:
                f.write(f"    OBSERVATIONS: {st['observations']}\n")
            if st["error"]:
                f.write(f"    ERROR: {st['error']}\n")
        f.write("\n")


# --------------------------------------------------------------------------- #
#  Main loop                                                                   #
# --------------------------------------------------------------------------- #
TASK_TIMEOUT_S = 240   # 4 minutes max per (task x run)
INTER_TASK_SLEEP = 3   # seconds between tasks to avoid rate limits


def run_one(task, model_id, prompt_variant, effort, temperature, run_index,
            search_backend="serper"):
    record = {
        "schema_version": "gt-pilot-2",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": task["task_id"],
        "model": model_id,
        "prompt_variant": prompt_variant,  # crossed facet: scaffold tool/code
        "effort": effort,              # crossed facet: reasoning_effort low/high
        "temperature": temperature,    # held constant at 1 (reasoning locks temp)
        "search_backend": search_backend,  # held constant; recorded for audit
        "task_type": coarse_task_type(task),
        "level": task["level"],
        "horizon_steps": task["horizon_steps"],
        "horizon_tools": task["horizon_tools"],
        "run_index": run_index,
        "question": task["question"],
        "true_answer": task["true_answer"],
        "has_attachment": int(bool(task["file_name"])),
        "timed_out": 0,
        "model_answer": None,
        "success": 0,
        "wall_time_s": None,
        "n_steps": None,
        "n_tool_calls": None,
        "error_flags": [],
        "trajectory": [],
        "crash": None,
    }
    t0 = time.time()
    try:
        agent = build_agent(model_id, prompt_variant, effort, temperature, search_backend)
        prompt = f"{GAIA_SYSTEM_HINT}\n\nTask: {task['question']}"
        if task["file_name"]:
            prompt += (
                f"\n\nNOTE: this task references an attached file "
                f"'{task['file_name']}'. Make sure the file is available locally; "
                f"this harness does not auto-download GAIA attachments."
            )
        raw = _run_with_timeout(agent.run, TASK_TIMEOUT_S, prompt)
        answer = extract_final_answer(raw)
        traj = serialize_trajectory(agent)
        record["model_answer"] = answer
        record["success"] = int(gaia_question_scorer(answer, task["true_answer"]))
        record["trajectory"] = traj
        record["n_steps"] = len(traj)
        record["n_tool_calls"] = sum(len(s["tool_calls"]) for s in traj)
        record["error_flags"] = error_flags(traj, record["success"], answer)
    except Exception as e:
        record["crash"] = f"{type(e).__name__}: {e}"
        record["timed_out"] = int(isinstance(e, TimeoutError))
        record["error_flags"] = ["harness_exception"]
        traceback.print_exc(file=sys.stderr)
    record["wall_time_s"] = round(time.time() - t0, 2)
    return record


def main():
    ap = argparse.ArgumentParser(description="GAIA GT pilot harness")
    ap.add_argument("--tasks", required=True, help="Path to gaia_level1.json")
    ap.add_argument("--model", required=True, help="LiteLLM model id")
    ap.add_argument("--runs", type=int, default=1, help="Replications per design cell")
    ap.add_argument("--efforts", default="low,high",
                    help="Comma-separated reasoning_effort levels (crossed facet). "
                         "Default: 'low,high'.")
    ap.add_argument("--scaffolds", default="tool,code",
                    help="Comma-separated manager scaffolds (crossed facet); each is "
                         "'tool' (ToolCallingAgent) or 'code' (CodeAgent). Populates "
                         "the 'prompt_variant' field. Default: 'tool,code'.")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="Sampling temperature, held CONSTANT across the design. "
                         "Must be 1.0 for gpt-5.x while reasoning_effort is low/high "
                         "(reasoning models reject temperature!=1). Default: 1.0.")
    ap.add_argument("--search", default="serper", choices=["serper", "tavily", "ddg"],
                    help="Web-search backend (held constant across the design). "
                         "Default: 'serper'.")
    ap.add_argument("--limit", type=int, default=None, help="Only first N tasks")
    ap.add_argument("--exclude-attachments", action="store_true",
                    help="Drop tasks that reference an attached file. The harness "
                         "never supplies attachments, so these tasks can only be "
                         "solved by guessing or by finding leaked answers online.")
    ap.add_argument("--shuffle", action="store_true",
                    help="Randomize the order of all (task x effort x scaffold x run) "
                         "units so that time-of-run is not confounded with condition.")
    ap.add_argument("--seed", type=int, default=2026, help="Seed for --shuffle.")
    ap.add_argument("--out", default="gaia_gt_runs.jsonl", help="Output JSONL path")
    ap.add_argument("--txt-dir", default=None,
                    help="If set, also write one human-readable .txt per task "
                         "(all iterations grouped) into this directory.")
    args = ap.parse_args()

    tasks = load_tasks(args.tasks)
    if args.exclude_attachments:
        tasks = [t for t in tasks if not t["file_name"]]
    if args.limit:
        tasks = tasks[: args.limit]

    # Crossed design facets. Every (effort x scaffold) combination is run on
    # every task for every replication -> a fully crossed task x effort x
    # scaffold x run design (the structure a G-study needs to decompose variance
    # cleanly across these facets). Temperature is held constant: gpt-5.x reasoning
    # models lock temperature to 1 whenever reasoning_effort is low/high, so it
    # cannot be a manipulated factor here -- the scaffold replaces it.
    efforts = [e.strip() for e in args.efforts.split(",") if e.strip()]
    scaffolds = [s.strip() for s in args.scaffolds.split(",") if s.strip()]
    bad = [s for s in scaffolds if s not in ("tool", "code")]
    if bad:
        ap.error(f"--scaffolds values must be 'tool' or 'code'; got {bad}")
    n_cells = len(tasks) * len(efforts) * len(scaffolds) * args.runs

    print(
        f"[pilot] {len(tasks)} tasks x {len(efforts)} efforts x "
        f"{len(scaffolds)} scaffolds x {args.runs} runs = {n_cells} cells"
        f" | model={args.model} | temp={args.temperature} (held)"
        f" | efforts={efforts} | scaffolds={scaffolds} | search={args.search}"
        f" | timeout={TASK_TIMEOUT_S}s/task",
        file=sys.stderr,
    )

    # Default order is nested (task > effort > scaffold > replication), which is how
    # the 2026-06-26 run was executed. --shuffle interleaves all units instead.
    units = [(t, e, s, r) for t in tasks for e in efforts for s in scaffolds
             for r in range(args.runs)]
    if args.shuffle:
        import random
        random.Random(args.seed).shuffle(units)

    n_done = n_ok = 0
    prev_task = None
    with open(args.out, "a", encoding="utf-8") as out:
        for t, effort, scaffold, r in units:
            # Brief pause whenever the task changes, to stay within API rate limits.
            if prev_task is not None and t["task_id"] != prev_task:
                time.sleep(INTER_TASK_SLEEP)
            prev_task = t["task_id"]
            rec = run_one(t, args.model, scaffold,
                          effort, args.temperature, r, args.search)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            if args.txt_dir:
                append_txt_block(args.txt_dir, rec)
            n_done += 1
            n_ok += rec["success"]
            status = "OK " if rec["success"] else ("ERR" if rec["crash"] else "miss")
            print(
                f"[{n_done:>4}/{n_cells}] {status}"
                f" task={t['task_id'][:8]}"
                f" e={effort} s={scaffold}"
                f" steps~{t['horizon_steps']}"
                f" t={rec['wall_time_s']}s",
                file=sys.stderr,
            )

    rate = (n_ok / n_done * 100) if n_done else 0.0
    print(f"\n[pilot] done. success {n_ok}/{n_done} = {rate:.1f}%  -> {args.out}", file=sys.stderr)
    if rate >= 80:
        print("[pilot] WARNING: >=80% -> ceiling effect likely.", file=sys.stderr)
    elif rate < 40:
        print("[pilot] WARNING: <40% -> floor effect likely.", file=sys.stderr)
    else:
        print("[pilot] success rate inside 40-80% estimable band. Good.", file=sys.stderr)


if __name__ == "__main__":
    main()
