import json
import sys
import requests
import wikienv

# ----------------------------------------------------------------------
# 1. Ollama LLM Caller (Replacing the paper's OpenAI call)
# ----------------------------------------------------------------------
def llm(prompt, stop=None):
    payload = {
        "model": "qwen2.5:3b",
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0}
    }
    if stop:
        payload["options"]["stop"] = stop if isinstance(stop, list) else [stop]
    resp = requests.post("http://localhost:11434/api/generate", json=payload, timeout=60)
    return resp.json().get("response", "").strip()

# ----------------------------------------------------------------------
# 2. Environment & Step Helper (From the paper's notebook)
# ----------------------------------------------------------------------
env = wikienv.WikiEnv()

def step(env, action):
    attempts = 0
    while attempts < 10:
        try:
            return env.step(action)
        except requests.exceptions.Timeout:
            attempts += 1
    return "Timeout", 0, True, {}

# ----------------------------------------------------------------------
# 3. Load Prompts
# ----------------------------------------------------------------------
folder = './prompts/'
prompt_file = 'prompts_naive.json'
with open(folder + prompt_file, 'r', encoding='utf-8') as f:
    prompt_dict = json.load(f)

webthink_examples = prompt_dict['webthink_simple6']
instruction = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
Here are some examples.
"""
webthink_prompt = instruction + webthink_examples

# ----------------------------------------------------------------------
# 4. The Core ReAct Function: webthink() (Verbatim from hotpotqa.ipynb)
# ----------------------------------------------------------------------
def webthink(question, prompt=webthink_prompt, to_print=True):
    _ = env.reset()
    if to_print:
        print(f"Question: {question}\n")
    prompt += f"Question: {question}\n"
    n_calls, n_badcalls = 0, 0
    for i in range(1, 8):
        n_calls += 1
        thought_action = llm(prompt + f"Thought {i}:", stop=[f"\nObservation {i}:"])
        try:
            thought, action = thought_action.strip().split(f"\nAction {i}: ")
        except:
            n_badcalls += 1
            n_calls += 1
            thought = thought_action.strip().split('\n')[0]
            action = llm(prompt + f"Thought {i}: {thought}\nAction {i}:", stop=[f"\n"]).strip()
        obs, r, done, info = step(env, action[0].lower() + action[1:])
        obs = str(obs).replace('\n', '')
        step_str = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
        prompt += step_str
        if to_print:
            print(step_str)
        if done:
            break
    if not done:
        obs, r, done, info = step(env, "finish[]")
    
    print("=" * 60)
    print(f"Final Answer: {env.answer}")
    print("=" * 60)
    return env.answer

# ----------------------------------------------------------------------
# 5. Run an example
# ----------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?"
    webthink(question)

