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

def load_sft_model(path, device, expected_vocab_size): 
    """Loads the pre-trained SFT model and handles vocabulary resizing."""
    print(f"Loading SFT model from {path} on {device}...")
    try:
        # Load the checkpoint dictionary
        ckpt = torch.load(path, map_location=device)
    except FileNotFoundError:
        print(f"ERROR: SFT model not found at {path}. Have you run python sft/train.py or fixed the file path?")
        raise
    
    # 1. Unwrap the state dictionary (Handles '_orig_mod.' prefix)
    state_dict = ckpt['model']
    unwrapped_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            unwrapped_state_dict[k[len('_orig_mod.'):]] = v
        else:
            unwrapped_state_dict[k] = v

    # 2. Override the vocab size in the config to match the tokenizer (50257 for gpt2)
    gptconf = GPTConfig(**ckpt['model_args'])
    
    old_vocab_size = gptconf.vocab_size
    if old_vocab_size != expected_vocab_size:
        print(f"Warning: Model vocab size ({old_vocab_size}) corrected to match tokenizer's size: {expected_vocab_size}.")
        gptconf.vocab_size = expected_vocab_size 

    # 3. Initialize the model with the large vocab size
    model = GPT(gptconf)
    
    # 4. Manually handle the size mismatch for embedding and head layers
    temp_state_dict = {}
    
    for k, v in unwrapped_state_dict.items():
        if k in ['transformer.wte.weight', 'lm_head.weight']:
            # Mismatched layer detected (old_vocab_size vs expected_vocab_size)
            
            # Get the new, larger tensor from the model (size expected_vocab_size)
            new_v = model.state_dict()[k].clone()
            
            # Copy the old (smaller) weights to the top part of the new tensor
            new_v[:old_vocab_size].copy_(v)
            temp_state_dict[k] = new_v
            print(f"INFO: Successfully copied old weights for {k} into the expanded model.")
        else:
            # All other layers (sizes should match)
            temp_state_dict[k] = v

    # Load the fixed state dict (must be strict=True now that sizes match)
    model.load_state_dict(temp_state_dict, strict=True)
    
    model.to(device)
    model.eval()
    print("SFT model loaded successfully.")
    return model


def encode(s): return [stoi[c] for c in s if c in stoi]
def decode(l): return ''.join([itos[i] for i in l])

def generate_negative_response(problem, model, encode, decode, device, max_tokens=64):
    """Uses the SFT model to generate a typically incorrect/default response."""
    input_ids = torch.tensor([encode(problem)], dtype=torch.long, device=device)
    with torch.no_grad():
        output_ids, _ = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=0.001,
            top_k=50
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
    match_algebra = re.match(r"(\d+)([+\-])x=(\d+), x=\?", problem)
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
        model = load_sft_model(SFT_MODEL_PATH, DEVICE, len(stoi)) 
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
        if positive_response:
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