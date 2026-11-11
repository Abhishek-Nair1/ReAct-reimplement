import os
import sys
import json
import requests
import wikienv

# ==============================================================================
# Block 1: LLM Caller (Ollama with local Qwen 2.5 3B)
# ==============================================================================
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")

def llm(prompt: str, stop=None) -> str:
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
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json().get("response", "")
    except Exception as e:
        print(f"[Error calling Ollama]: {e}")
        return ""

# ==============================================================================
# Block 2: Environment Setup (WikiEnv for FEVER)
# ==============================================================================
env = wikienv.WikiEnv()

def step(env_instance, action: str):
    attempts = 0
    while attempts < 10:
        try:
            return env_instance.step(action)
        except requests.exceptions.Timeout:
            attempts += 1
    return "Timeout connecting to Wikipedia", 0, True, {}

# ==============================================================================
# Block 3: Prompt Loading & ReAct Loop
# ==============================================================================
with open("./prompts/fever.json", "r", encoding="utf-8") as f:
    prompt_dict = json.load(f)

webthink_prompt = prompt_dict["webthink_simple3"]

def webthink_fever(claim: str, prompt: str = webthink_prompt, to_print: bool = True):
    claim_str = f"Claim: {claim}"
    if to_print:
        print(claim_str + "\n")
    
    prompt += claim_str + "\n"
    n_calls, n_badcalls = 0, 0

    for i in range(1, 8):
        n_calls += 1
        thought_action = llm(prompt + f"Thought {i}:", stop=[f"\nObservation {i}:"])
        
        try:
            thought, action = thought_action.strip().split(f"\nAction {i}: ")
            action = action.split("\n")[0].strip()
        except Exception:
            n_badcalls += 1
            n_calls += 1
            thought = thought_action.strip().split("\n")[0]
            action = llm(prompt + f"Thought {i}: {thought}\nAction {i}:", stop=["\n"]).strip()
            action = action.split("\n")[0].strip()

        obs, r, done, info = step(env, action[0].lower() + action[1:])
        obs = str(obs).replace("\n", " ").strip()
        step_str = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
        prompt += step_str

        if to_print:
            print(step_str)

        if done:
            break

    if not done:
        obs, r, done, info = step(env, "finish[]")

    return env.answer

# ==============================================================================
# Block 4: Execution / Evaluation
# ==============================================================================
if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_claim = " ".join(sys.argv[1:])
    else:
        test_claim = "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company."
    ans = webthink_fever(test_claim)
    print("=" * 60)
    print(f"Fact Verification Result: {ans}")
    print("=" * 60)
