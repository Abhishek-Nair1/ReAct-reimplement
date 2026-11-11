import os
import sys
import json
import time
import random
import re
import string
import argparse
from collections import Counter
import requests
import wikienv

# ==============================================================================
# Configuration & LLM Caller (Ollama with local Qwen 2.5 3B)
# ==============================================================================
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")

def llm(prompt: str, stop=None, retries: int = 3) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "top_p": 1.0,
            "num_predict": 128,
            "stop": stop or ["\n"]
        }
    }
    for attempt in range(retries):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=90)
            resp.raise_for_status()
            text = resp.json().get("response", "")
            if text:
                return text
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                print(f"[LLM Error after {retries} retries]: {e}", flush=True)
    return ""

# ==============================================================================
# Metric Normalization & Scoring (From official ReAct wrappers.py)
# ==============================================================================
def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))

def compute_f1(prediction: str, ground_truth: str) -> float:
    norm_pred = normalize_answer(prediction)
    norm_gt = normalize_answer(ground_truth)

    if norm_pred in ["yes", "no", "noanswer"] and norm_pred != norm_gt:
        return 0.0
    if norm_gt in ["yes", "no", "noanswer"] and norm_pred != norm_gt:
        return 0.0

    pred_tokens = norm_pred.split()
    gt_tokens = norm_gt.split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0 or len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0

    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)

def compute_em(prediction: str, ground_truth: str) -> int:
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))

# ==============================================================================
# Wikipedia Environment Step Helper
# ==============================================================================
env = wikienv.WikiEnv()

def step_env(action: str):
    for _ in range(5):
        try:
            time.sleep(0.2)  # courteous delay to prevent rate limits
            act_formatted = action[0].lower() + action[1:] if action else ""
            res = env.step(act_formatted)
            obs = res[0]
            done = res[2] if len(res) == 4 else res[1]
            return obs, done
        except requests.exceptions.Timeout:
            time.sleep(2)
        except Exception:
            time.sleep(1)
    return "Timeout connecting to Wikipedia", True

# ==============================================================================
# ReAct Loop for HotpotQA & FEVER
# ==============================================================================
def run_react_sample(task: str, query: str, prompt_template: str, max_steps: int = 7, verbose: bool = True):
    prefix = "Question: " if task == "hotpotqa" else "Claim: "
    query_str = f"{prefix}{query}\n"
    current_prompt = prompt_template + query_str
    
    if verbose:
        print(f"\n[{task.upper()}] {query_str.strip()}", flush=True)

    trajectory = []
    env.reset()

    for i in range(1, max_steps + 1):
        thought_action = llm(current_prompt + f"Thought {i}:", stop=[f"\nObservation {i}:"])
        
        try:
            thought, action = thought_action.strip().split(f"\nAction {i}: ")
            action = action.split("\n")[0].strip()
        except Exception:
            thought = thought_action.strip().split("\n")[0]
            action = llm(current_prompt + f"Thought {i}: {thought}\nAction {i}:", stop=["\n"]).strip()
            action = action.split("\n")[0].strip()

        obs, done = step_env(action)
        obs_clean = str(obs).replace("\n", " ").strip()
        step_text = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs_clean}\n"
        current_prompt += step_text

        trajectory.append({
            "step": i,
            "thought": thought,
            "action": action,
            "observation": obs_clean
        })

        if verbose:
            print(f"  Thought {i}: {thought}", flush=True)
            print(f"  Action {i}: {action}", flush=True)
            print(f"  Observation {i}: {obs_clean[:120]}..." if len(obs_clean) > 120 else f"  Observation {i}: {obs_clean}", flush=True)

        if done:
            break

    if not env.answer:
        _, _ = step_env("Finish[]")

    prediction = env.answer or ""
    return prediction, trajectory

# ==============================================================================
# Evaluation Pipeline
# ==============================================================================
def evaluate(task: str, count: int, start: int = 0, seed: int = 233, verbose: bool = True, output_file: str = None):
    # 1. Load Prompts
    if task == "hotpotqa":
        with open("./prompts/prompts_naive.json", "r", encoding="utf-8") as f:
            pdict = json.load(f)
        instruction = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
Here are some examples.
"""
        prompt_template = instruction + pdict["webthink_simple6"]

        data_path = "./data/hotpot_dev_v1_simplified.json"
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        samples = [(d["question"], d["answer"]) for d in raw_data]

    elif task == "fever":
        with open("./prompts/fever.json", "r", encoding="utf-8") as f:
            pdict = json.load(f)
        prompt_template = pdict["webthink_simple3"]

        data_path = "./data/paper_dev.jsonl"
        samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                samples.append((item["claim"], item["label"]))
    else:
        raise ValueError(f"Unknown task: {task}. Must be 'hotpotqa' or 'fever'.")

    # 2. Reproducible Sampling (Matching paper: seed 233)
    indices = list(range(len(samples)))
    random.Random(seed).shuffle(indices)
    target_indices = indices[start : start + count]

    # 3. Check for existing results to resume cleanly
    results = []
    em_list = []
    f1_list = []
    completed_indices = set()
    prior_elapsed = 0.0

    if output_file and os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f_in:
                existing_data = json.load(f_in)
                if existing_data.get("task") == task:
                    for r in existing_data.get("results", []):
                        if r["index"] in target_indices:
                            results.append(r)
                            em_list.append(r["em"])
                            f1_list.append(r["f1"])
                            completed_indices.add(r["index"])
                    if results:
                        prior_elapsed = existing_data.get("elapsed_seconds", 0.0)
                        print(f"[*] Resuming from existing results in {output_file}: {len(results)} samples already completed.", flush=True)
        except Exception as e:
            print(f"[Warning] Could not load existing results for resuming: {e}", flush=True)

    print("=" * 70, flush=True)
    print(f"Starting ReAct Evaluation | Task: {task.upper()} | Samples Target: {len(target_indices)} | Seed: {seed}", flush=True)
    print(f"Model: {OLLAMA_MODEL} | Dataset Size: {len(samples)} | Already Completed: {len(results)}", flush=True)
    print("=" * 70, flush=True)

    start_time = time.time()

    for idx_num, sample_idx in enumerate(target_indices, start=1):
        if sample_idx in completed_indices:
            continue

        query, ground_truth = samples[sample_idx]
        t0 = time.time()

        pred, trajectory = run_react_sample(task, query, prompt_template, verbose=verbose)
        sample_time = time.time() - t0

        em = compute_em(pred, ground_truth)
        f1 = compute_f1(pred, ground_truth)

        em_list.append(em)
        f1_list.append(f1)

        result_item = {
            "index": sample_idx,
            "query": query,
            "ground_truth": ground_truth,
            "prediction": pred,
            "em": em,
            "f1": f1,
            "seconds": round(sample_time, 2),
            "trajectory": trajectory
        }
        results.append(result_item)

        avg_em = sum(em_list) / len(em_list)
        avg_f1 = sum(f1_list) / len(f1_list)
        print(f"\n--> Result [{len(results)}/{len(target_indices)}]:", flush=True)
        print(f"    Prediction:   {pred}", flush=True)
        print(f"    Ground Truth: {ground_truth}", flush=True)
        print(f"    Metrics:      EM = {em} | F1 = {f1:.3f} (Time: {sample_time:.1f}s)", flush=True)
        print(f"    Cumulative:   Avg EM = {avg_em:.3f} | Avg F1 = {avg_f1:.3f}", flush=True)
        print("-" * 70, flush=True)

        if output_file:
            os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f_out:
                json.dump({
                    "task": task,
                    "model": OLLAMA_MODEL,
                    "count": len(results),
                    "avg_em": avg_em,
                    "avg_f1": avg_f1,
                    "elapsed_seconds": round(prior_elapsed + (time.time() - start_time), 2),
                    "results": results
                }, f_out, indent=2)

    total_time = prior_elapsed + (time.time() - start_time)
    print("\n" + "=" * 70, flush=True)
    print(f"EVALUATION COMPLETE ({task.upper()})", flush=True)
    print(f"Total Evaluated: {len(results)}", flush=True)
    print(f"Mean Exact Match (EM): {sum(em_list)/len(em_list):.4f}", flush=True)
    if task == "hotpotqa":
        print(f"Mean F1 Score:         {sum(f1_list)/len(f1_list):.4f}", flush=True)
    print(f"Total Wall Time:       {total_time:.1f}s", flush=True)
    if output_file:
        print(f"Results Saved To:      {output_file}", flush=True)
    print("=" * 70, flush=True)

# ==============================================================================
# CLI Entrypoint
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ReAct on HotpotQA or FEVER")
    parser.add_argument("--task", type=str, default="hotpotqa", choices=["hotpotqa", "fever"], help="Benchmark task")
    parser.add_argument("--count", type=int, default=5, help="Number of samples to evaluate (default: 5)")
    parser.add_argument("--start", type=int, default=0, help="Start offset in sampled indices")
    parser.add_argument("--seed", type=int, default=233, help="Random seed (paper default: 233)")
    parser.add_argument("--output", type=str, default=None, help="Path to save JSON results")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode (suppress step traces)")

    args = parser.parse_args()
    
    out_path = args.output or f"results_{args.task}.json"
    evaluate(
        task=args.task,
        count=args.count,
        start=args.start,
        seed=args.seed,
        verbose=not args.quiet,
        output_file=out_path
    )
