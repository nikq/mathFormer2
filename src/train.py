import torch
import torch.optim as optim
import threading
import time
import numpy as np
from src.data.generator import MathGenerator
from src.model.tokenizer import CharTokenizer
from src.model.transformer import MathFormer, MathFormerConfig
from src.vis.server import start_server, update_state

# Hyperparameters
BATCH_SIZE = 64
MAX_ITERS = 5000
LEARNING_RATE = 3e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.backends.mps.is_available():
    DEVICE = 'mps'

def train():
    print(f"Using device: {DEVICE}")
    
    # 1. Setup Data
    generator = MathGenerator()
    tokenizer = CharTokenizer()
    
    # 2. Setup Model
    config = MathFormerConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=128,
        n_layer=4,
        n_head=4,
        d_model=128
    )
    model = MathFormer(config).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    print("Model initialized.")

    # 3. Training Loop
    model.train()
    for iter_num in range(MAX_ITERS):
        # Generate batch
        # Difficulty schedule: increase every 1000 iters
        difficulty = 1 + (iter_num // 1000)
        batch_strs = generator.generate_batch(BATCH_SIZE, difficulty=difficulty)
        
        # Tokenize
        # We need to pad to the longest in the batch or fixed block_size
        # For simplicity, let's just pad to block_size or truncate
        max_len = config.block_size
        X_batch = []
        Y_batch = []
        
        for s in batch_strs:
            ids = tokenizer.encode(s)
            # Truncate if too long
            if len(ids) > max_len:
                ids = ids[:max_len]
            
            # Pad
            pad_len = max_len - len(ids)
            padded_ids = ids + [tokenizer.pad_token_id] * pad_len
            
            # For causal prediction: input is ids[:-1], target is ids[1:]
            # But here we want to predict the next token.
            # Standard GPT training:
            # Input:  [A, B, C, PAD]
            # Target: [B, C, PAD, PAD] (ignore loss on PAD)
            
            # Actually, let's just train on the full sequence shifted
            # x = [0, 1, 2] -> y = [1, 2, 3]
            
            # We need to handle padding carefully in loss, but for toy model:
            # Just mask out padding in loss if possible, or let it learn to predict pad after pad.
            
            X_batch.append(padded_ids[:-1])
            Y_batch.append(padded_ids[1:])
            
        X = torch.tensor(X_batch, dtype=torch.long).to(DEVICE)
        Y = torch.tensor(Y_batch, dtype=torch.long).to(DEVICE)
        
        # Forward
        logits, loss = model(X, Y)
        
        # Backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        # Visualization Update (every 10 steps)
        if iter_num % 10 == 0:
            loss_val = loss.item()
            print(f"Iter {iter_num}: Loss {loss_val:.4f}, Difficulty {difficulty}")
            
            # Extract weights
            # 1. All model weights
            # 2. Attention weights (dynamic)
            
            with torch.no_grad():
                # Attention from last layer, first head, first sample
                # shape: [B, H, T, T]
                attn_weights = model.get_attention_weights()[-1][0, 0, :, :].cpu().numpy()
                
                # All learnable weights
                all_weights = model.get_all_weights()
                
                # Send to server
                vis_data = {
                    "layer_weights": all_weights, # Now a dict of all weights
                    "attention": attn_weights.tolist()
                }
                update_state(vis_data, iter_num, loss_val)
                
        if iter_num % 100 == 0:
            # Generate a sample to see progress
            model.eval()
            with torch.no_grad():
                # Prompt: "1+1="
                prompt = "10+10="
                x = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(DEVICE)
                # Generate a few tokens
                for _ in range(10):
                    logits, _ = model(x)
                    # Greedy decode
                    next_token = logits[0, -1, :].argmax()
                    x = torch.cat((x, next_token.unsqueeze(0).unsqueeze(0)), dim=1)
                    if next_token.item() == tokenizer.pad_token_id:
                        break
                
                decoded = tokenizer.decode(x[0].tolist())
                print(f"Sample: {decoded}")
            model.train()

if __name__ == "__main__":
    # Start Server in background thread
    server_thread = threading.Thread(target=start_server, kwargs={'port': 8000}, daemon=True)
    server_thread.start()
    
    print("Server started at http://localhost:8000")
    
    # Start Training
    try:
        train()
    except KeyboardInterrupt:
        print("Training stopped.")
