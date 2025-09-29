import pickle
import random
import json
import re
import torch
from transformers import AutoTokenizer
# Assuming your NanoGPT model definition is accessible via these imports
from model import GPT, GPTConfig 

# --- CONFIGURATION ---
NUM_PROBLEMS_TO_GENERATE = 100 
OUTPUT_FILE = "data/pos_neg_pairs.json"
SFT_MODEL_PATH = "sft/gpt.pt"
# Check if CUDA is available, otherwise use CPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

with open("sft/meta.pkl", "rb") as f:
    meta = pickle.load(f)
stoi, itos = meta["stoi"], meta["itos"]

# --- 1. PROBLEM GENERATION FUNCTIONS (Task 1: Step 1) ---

def generate_arithmetic_problem():
    """Generates a random arithmetic problem (A op B=? with integer results for division)."""
    op = random.choice(['+', '-', '*', '/'])
    
    if op in ['+', '-']:
        A = random.randint(10, 100)
        B = random.randint(10, 100)
        if op == '-':
            A, B = max(A, B), min(A, B)
    
    elif op == '*':
        A = random.randint(2, 20)
        B = random.randint(2, 20)
    
    elif op == '/':
        result = random.randint(2, 10)
        B = random.randint(2, 10)
        A = result * B 

    return f"{A}{op}{B}=?"

def generate_algebra_problem():
    """Generates a simple linear algebra problem (A op x = C, x=?)."""
    op = random.choice(['+', '-'])
    X = random.randint(1, 50)
    B = random.randint(10, 100)
    
    if op == '+':
        C = X + B
        query = f"{B}+x={C}, x=?"
    else:
        C = B - X
        query = f"{B}-x={C}, x=?"
        
    return query

def generate_raw_problems(n):
    """Generates the main list of problems."""
    raw_problems = []
    for _ in range(n):
        if random.random() < 0.6: 
            problem = generate_arithmetic_problem()
        else:
            problem = generate_algebra_problem()
        raw_problems.append(problem)
    return raw_problems

# --- 2. NEGATIVE RESPONSE GENERATOR (Task 1: Step 2) ---

def load_sft_model(path, device, ):
    """Loads the pre-trained SFT model using a direct checkpoint loading approach."""
    print(f"Loading SFT model from {path} on {device}...")
    try:
        ckpt = torch.load(path, map_location=device)
    except FileNotFoundError:
        print(f"ERROR: SFT model not found at {path}. Have you run python sft/train.py or fixed the file path?")
        raise

    gptconf = GPTConfig(**ckpt['model_args'])
    gpt = GPT(gptconf)
    state_dict = ckpt['model']
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    gpt.load_state_dict(state_dict)
    gpt.to(device).train()
    
    print("SFT model loaded successfully.")
    return gpt


def encode(s): return [stoi[c] for c in s if c in stoi]
def decode(l): return ''.join([itos[i] for i in l])

def generate_negative_response(problem, model, encode, decode, device, max_tokens=200):
    """Uses the SFT model to generate a typically incorrect/default response."""
    input_ids = torch.tensor([encode(problem)], dtype=torch.long, device=device)
    with torch.no_grad():
        output_ids, _ = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=0.8,
            top_k=200

        )
    response = decode(output_ids[0].tolist())
    negative_text = response[len(problem):].strip()
    #if not negative_text or len(negative_text) < 5:
        #negative_text = "Sorry, I don't know the answer to this math problem."
    return problem + negative_text

# --- 3. POSITIVE RESPONSE GENERATOR (Task 1: Step 3) ---

def solve_and_format_positive_response(problem):
    """Solves the problem and formats the positive response with explanation."""
    
    def safe_eval_arithmetic(expr):
        try:
            clean_expr = expr.replace('$', '').replace('?', '').replace('=', '').strip()
            # Note: eval() is generally unsafe, but acceptable here for simple arithmetic problems in a private script context.
            result = eval(clean_expr)
            return str(int(result)) if result == int(result) else str(result)
        except:
            return None

    # ARITHMETIC PROBLEM HANDLING: Pattern for $A op B=?$
    match_arithmetic = re.match(r"([\d\.]+)([+\-*/])([\d\.]+)=\?", problem)
    if match_arithmetic:
        A, op, B = match_arithmetic.groups()
        expression = f"{A}{op}{B}"
        answer = safe_eval_arithmetic(expression)
        if answer is not None:
            return f"{problem} The answer is {answer} because {expression} equals {answer}."

    # ALGEBRA PROBLEM HANDLING: Pattern for $A op x = C, x=?$
    match_algebra = re.match(r"(\d+)([+\-])x=(\-?\d+), x=\?", problem)
    if match_algebra:
        A, op, C = match_algebra.groups()
        A, C = int(A), int(C)
        if op == '+': # x = C - A
            x = C - A
            return f"{problem} The answer is {x} because {C} minus {A} equals {x}."
        elif op == '-': # x = A - C
            x = A - C
            return f"{problem} The answer is {x} because {A} minus {C} equals {x}."

    return None 

# --- 4. MAIN EXECUTION AND SAVING (Task 1: Step 4) ---

def main():
    
    # --- Load necessary components ---

    
    try:
        model = load_sft_model(SFT_MODEL_PATH, DEVICE) 
    except FileNotFoundError:
        print("\n*** ACTION REQUIRED: Please ensure sft/gpt.pt exists and try again. ***")
        return
    except Exception as e:
        print(f"\n*** FATAL ERROR during model loading: {e} ***")
        return


    # --- Step 1: Generate Raw Problems ---
    print(f"\nGenerating {NUM_PROBLEMS_TO_GENERATE} raw math problems...")
    raw_problems = generate_raw_problems(NUM_PROBLEMS_TO_GENERATE)
    
    # --- Step 2 & 3: Generate Negative and Positive Pairs ---
    dpo_dataset = []
    
    for i, problem in enumerate(raw_problems):
        if i % 1000 == 0:
            print(f"Processing problem {i}/{NUM_PROBLEMS_TO_GENERATE}...")

        positive_response = solve_and_format_positive_response(problem)
        #if positive_response:
        negative_response = generate_negative_response(problem, model, encode, decode, DEVICE)
        dpo_dataset.append({
            "negative": negative_response,
            "positive": positive_response
        })

    # --- Step 4: Save Data ---
    print(f"\nSuccessfully generated {len(dpo_dataset)} complete pairs.")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(dpo_dataset, f, indent=4)

    print(f"Data saved to {OUTPUT_FILE}.")
    print("\nTask 1 (Data Preparation) is complete! You can now proceed to the DPO training notebook (dpo/dpo.ipynb).")

if __name__ == "__main__":
    main()