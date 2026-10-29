"""Prompts, strict parser, feedback, resumable L0-L3 loop for precedence TSP."""
import json
import re
import time
import datetime
from tsp_utils import check_tour, tour_length
from cpsat_utils import validate_with_cpsat

K_ATTEMPTS = 5  # total attempts per feedback level (attempt 1 = the L1 answer)


# ---------------------------------------------------------------- prompts
def _coord_lines(inst):
    return "\n".join(f"{i}: ({x:.4f}, {y:.4f})" for i, (x, y) in enumerate(inst["coordinates"]))


def prompt_l0(inst):
    n = inst["n_cities"]
    rules = "; ".join(f"city {a} before city {b}" for a, b in inst["precedences"])
    return (
        f"Solve this routing problem. There are {n} cities, numbered 0 to {n - 1}, "
        f"with coordinates (x, y):\n{_coord_lines(inst)}\n\n"
        f"Starting at city 0, visit every other city exactly once and then return to city 0, "
        f"keeping the total Euclidean distance as short as possible. "
        f"Some cities must be visited before others: {rules}.\n"
        f"Give the tour as a list of city numbers."
    )


def prompt_l1(inst):
    n = inst["n_cities"]
    rules = "\n".join(f"   - city {a} before city {b}" for a, b in inst["precedences"])
    return (
        "You are solving a Traveling Salesperson Problem with precedence constraints.\n\n"
        f"There are {n} cities, numbered 0 to {n - 1}. City 0 is the depot. "
        f"Coordinates (x, y):\n{_coord_lines(inst)}\n\n"
        "Rules. A tour that breaks any rule is invalid:\n"
        "1. The tour starts at city 0.\n"
        f"2. The tour visits each of the cities 1 to {n - 1} exactly once.\n"
        "3. The tour ends by returning to city 0. City 0 appears only as the first and the last entry.\n"
        f"4. The tour has exactly {n + 1} entries.\n"
        "5. Precedence: for every pair \"a before b\" below, city a must appear earlier "
        "in the tour than city b:\n"
        f"{rules}\n\n"
        "Objective: among valid tours, make the total Euclidean distance as short as possible.\n\n"
        "Output format: return ONLY a JSON object, with no other text: {\"tour\": [0, ..., 0]}"
    )


FEEDBACK_L2 = 'Your tour is invalid. Try again. Return only the JSON object {"tour": [0, ..., 0]}.'


def feedback_l3(inst, rec):
    """Specific feedback built from the checker's diagnosis of the previous reply."""
    n = inst["n_cities"]
    lines = []
    if rec["parsed_tour"] is None:
        lines.append("Your reply could not be read as a JSON object with a \"tour\" list of integers.")
    else:
        for e in rec["errors"]:
            t = e["type"]
            if t == "non_integer":
                lines.append("Every entry of the tour must be an integer city number.")
            elif t == "wrong_length":
                lines.append(f"The tour has {e['got']} entries but must have exactly {e['expected']}.")
            elif t == "wrong_start":
                lines.append("The tour must start at city 0.")
            elif t == "wrong_end":
                lines.append("The tour must end at city 0.")
            elif t == "out_of_range":
                lines.append(f"These city numbers do not exist: {e['cities']}. Valid cities are 0 to {n - 1}.")
            elif t == "missing":
                lines.append(f"These cities are never visited: {e['cities']}.")
            elif t == "duplicate":
                lines.append(f"These cities appear more than once (city 0 may only be first and last): {e['cities']}.")
            elif t == "precedence":
                for a, b in e["pairs"]:
                    lines.append(f"City {a} must be visited before city {b}, but your tour visits {b} first.")
    body = "\n".join(f"- {s}" for s in lines)
    return ("Your tour is invalid:\n" + body +
            "\nFix these problems and try again. Return only the JSON object {\"tour\": [0, ..., 0]}.")


# ----------------------------------------------------------------- parser
def parse_reply(text):
    """Strict. Returns (tour_or_None, parse_error_or_None). No repair of any kind."""
    if text is None or not text.strip():
        return None, "empty_reply"
    s = text.strip()
    m = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", s, flags=re.S | re.I)
    if m:
        s = m.group(1).strip()
    try:
        obj = json.loads(s)
    except Exception:
        return None, "not_json"
    if isinstance(obj, dict):
        if "tour" not in obj:
            return None, "no_tour_key"
        obj = obj["tour"]
    if not isinstance(obj, list):
        return None, "tour_not_a_list"
    return obj, None


# -------------------------------------------------------------- model call
class DailyQuotaExceeded(RuntimeError):
    pass


def make_gemini_caller(client, model_id, thinking_level, max_retries=6):
    """Returns caller(messages) -> dict. messages = [(role, text)], role in {'user','model'}."""
    from google.genai import types

    def caller(messages):
        contents = [types.Content(role=r, parts=[types.Part(text=t)]) for r, t in messages]
        cfg = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level=thinking_level))
        retries = 0
        for attempt in range(max_retries):
            try:
                t0 = time.perf_counter()
                resp = client.models.generate_content(model=model_id, contents=contents, config=cfg)
                latency = time.perf_counter() - t0
                u = resp.usage_metadata
                fr = None
                try:
                    fr = str(resp.candidates[0].finish_reason)
                except Exception:
                    pass
                return {"text": resp.text, "latency": latency, "retries": retries, "finish_reason": fr,
                        "input_tokens": getattr(u, "prompt_token_count", None),
                        "output_tokens": getattr(u, "candidates_token_count", None),
                        "thinking_tokens": getattr(u, "thoughts_token_count", None)}
            except Exception as e:
                msg = str(e)
                transient = any(k in msg for k in ("429", "RESOURCE_EXHAUSTED", "500", "503", "UNAVAILABLE", "DEADLINE"))
                if not transient:
                    raise
                if "PerDay" in msg or "per day" in msg.lower():
                    raise DailyQuotaExceeded(msg)
                retries += 1
                wait = min(90, 10 * 2 ** attempt)
                print(f"  transient API error ({msg[:80]}...); waiting {wait}s")
                time.sleep(wait)
        raise RuntimeError("API failed after retries")
    return caller


# ------------------------------------------------------------------ logging
class CallLog:
    def __init__(self, path):
        self.path = path
        self.records = {}
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        self.records[(r["instance_id"], r["level"], r["attempt"])] = r

    def get(self, key):
        return self.records.get(key)

    def add(self, rec):
        key = (rec["instance_id"], rec["level"], rec["attempt"])
        assert key not in self.records, key
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
        self.records[key] = rec


# -------------------------------------------------------------------- loop
class Runner:
    def __init__(self, caller, log, model_id, thinking_level, pause=5.0, max_new_calls=None):
        self.caller, self.log = caller, log
        self.model_id, self.thinking_level = model_id, thinking_level
        self.pause, self.max_new_calls = pause, max_new_calls
        self.new_calls = 0

    def call(self, inst, level, attempt, messages):
        key = (inst["instance_id"], level, attempt)
        cached = self.log.get(key)
        if cached is not None:
            return cached
        if self.max_new_calls is not None and self.new_calls >= self.max_new_calls:
            raise DailyQuotaExceeded("max_new_calls reached (self-imposed cap)")
        res = self.caller(messages)
        self.new_calls += 1
        n = inst["n_cities"]
        tour, perr = parse_reply(res["text"])
        if tour is None:
            chk = {"valid": False, "errors": [{"type": perr}]}
            cp_valid = False
        else:
            chk = check_tour(tour, n, inst["precedences"])
            cp_valid = validate_with_cpsat(inst, tour)
        rec = {
            "instance_id": inst["instance_id"], "n_cities": n, "level": level, "attempt": attempt,
            "user_message": messages[-1][1], "raw_reply": res["text"] or "",
            "parsed_tour": tour, "parse_error": perr,
            "py_valid": chk["valid"], "cpsat_valid": cp_valid,
            "error_types": [e["type"] for e in chk["errors"]], "errors": chk["errors"],
            "tour_length": tour_length(inst["coordinates"], tour) if chk["valid"] else None,
            "input_tokens": res.get("input_tokens"), "output_tokens": res.get("output_tokens"),
            "thinking_tokens": res.get("thinking_tokens"), "latency_seconds": res.get("latency"),
            "retries": res.get("retries"), "finish_reason": res.get("finish_reason"),
            "model_id": self.model_id, "thinking_level": self.thinking_level,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        if rec["py_valid"] != rec["cpsat_valid"]:
            print(f"  !! checker/CP-SAT DISAGREE on {key}")
        self.log.add(rec)
        time.sleep(self.pause)
        return rec

    @staticmethod
    def _history(chain):
        h = []
        for r in chain:
            h += [("user", r["user_message"]), ("model", r["raw_reply"] or "[empty reply]")]
        return h

    def run_instance(self, inst):
        self.call(inst, "L0", 1, [("user", prompt_l0(inst))])
        r1 = self.call(inst, "L1", 1, [("user", prompt_l1(inst))])
        for level in ("L2", "L3"):
            chain, last = [r1], r1
            for attempt in range(2, K_ATTEMPTS + 1):
                if last["py_valid"]:
                    break
                fb = FEEDBACK_L2 if level == "L2" else feedback_l3(inst, last)
                last = self.call(inst, level, attempt, self._history(chain) + [("user", fb)])
                chain.append(last)

    def summary_line(self, inst):
        iid, parts = inst["instance_id"], []
        for level in ("L0", "L1", "L2", "L3"):
            recs = [self.log.get((iid, level, a)) for a in range(1, K_ATTEMPTS + 1)]
            recs = [r for r in recs if r]
            if level in ("L2", "L3"):
                recs = [self.log.get((iid, "L1", 1))] + recs
            first_ok = next((i + 1 for i, r in enumerate(recs) if r["py_valid"]), None)
            parts.append(f"{level}:{'ok@' + str(first_ok) if first_ok else 'FAIL'}")
        return f"{iid}  " + "  ".join(parts)
