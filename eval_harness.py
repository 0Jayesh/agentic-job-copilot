"""
eval_harness.py
 
Golden-set evaluation for the LLM-driven parts of the pipeline
(parse_node -> score_node -> planner_node), per spec section 3.5.
 
This is deliberately NOT a pytest-style unit test and doesn't live under
tests/: it makes real Gemini/Groq API calls (via parse_node) and grades
against tolerance ranges rather than exact values, because LLM output
varies run to run. The deterministic parts (MCP tools, calendar logic,
scoring math, memory stores) already have exact-match unit tests under
tests/ -- conflating the two testing styles is exactly the mistake the
spec calls out avoiding.
 
Run directly:
    python eval_harness.py
 
Exits non-zero if any non-informational check fails, so it can gate CI
(see .github/workflows/eval.yml).
"""
import statistics
import time
 
from dotenv import load_dotenv
load_dotenv()
 
from nodes import parse_node, score_node, planner_node
from resume import RESUME_TEXT
from structured_memory import is_unresolved_company
from golden_set import GOLDEN_SET
 
 
def evaluate_case(case: dict) -> dict:
    start = time.time()
    parsed = parse_node({"raw_input": case["raw_input"]})
    scored = score_node(parsed, RESUME_TEXT)
    planned = planner_node(scored)
    elapsed = time.time() - start
 
    result = {
        "id": case["id"],
        "elapsed": elapsed,
        "checks": [],
        "passed": True,
        "parsed": parsed,
        "fit_score": scored.get("fit_score"),
        "needs_followup": planned.get("needs_followup"),
    }
 
    def check(label: str, condition: bool, informational: bool = False):
        result["checks"].append(
            {"label": label, "passed": bool(condition), "informational": informational}
        )
        if not condition and not informational:
            result["passed"] = False
 
    expected = case["expected"]
    company = parsed.get("company")
 
    if expected.get("expect_unresolved_company"):
        check("company correctly left unresolved (no hallucinated name)", is_unresolved_company(company))
    elif expected.get("company_contains"):
        target = expected["company_contains"].lower()
        check(
            f"company contains '{expected['company_contains']}' (got {company!r})",
            bool(company) and target in company.lower(),
        )
 
    if expected.get("role_contains"):
        role = parsed.get("role") or ""
        target = expected["role_contains"].lower()
        check(f"role contains '{expected['role_contains']}' (got {role!r})", target in role.lower())
 
    if expected.get("fit_score_min") is not None:
        lo, hi = expected["fit_score_min"], expected["fit_score_max"]
        fit_score = scored.get("fit_score")
        informational = expected.get("fit_score_informational", False)
        in_range = fit_score is not None and lo <= fit_score <= hi
        check(f"fit_score in [{lo}, {hi}] (got {fit_score})", in_range, informational=informational)
 
    if "expected_needs_followup" in expected:
        informational = expected.get("needs_followup_informational", False)
        actual = planned.get("needs_followup")
        check(
            f"needs_followup == {expected['expected_needs_followup']} (got {actual})",
            actual == expected["expected_needs_followup"],
            informational=informational,
        )
 
    return result
 
 
def run_eval(golden_set=None, verbose: bool = True) -> tuple:
    golden_set = golden_set if golden_set is not None else GOLDEN_SET
    results = []
 
    for case in golden_set:
        try:
            r = evaluate_case(case)
        except Exception as e:
            r = {
                "id": case["id"],
                "passed": False,
                "error": f"{type(e).__name__}: {e}",
                "checks": [],
                "elapsed": None,
            }
        results.append(r)
 
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    latencies = [r["elapsed"] for r in results if r.get("elapsed") is not None]
 
    if verbose:
        print("=" * 72)
        for r in results:
            status = "PASS" if r["passed"] else "FAIL"
            elapsed_str = f"{r['elapsed']:.2f}s" if r.get("elapsed") is not None else "n/a"
            print(f"[{status}] {r['id']} (elapsed: {elapsed_str})")
            if "error" in r:
                print(f"    ERROR: {r['error']}")
            for c in r.get("checks", []):
                tag = "info" if c["informational"] else ("ok " if c["passed"] else "FAIL")
                print(f"    [{tag}] {c['label']}")
        print("=" * 72)
        print(f"Golden-set eval: {passed}/{total} passed")
        if latencies:
            print(
                f"Latency -- avg: {statistics.mean(latencies):.2f}s, "
                f"max: {max(latencies):.2f}s, min: {min(latencies):.2f}s"
            )
        print("=" * 72)
 
    return passed, total, results
 
 
if __name__ == "__main__":
    passed, total, _ = run_eval()
    if passed < total:
        raise SystemExit(1)