import torch
import torch.optim as optim
import threading
import time
import numpy as np
from src.data.generator import MathGenerator
from src.model.tokenizer import CharTokenizer
from src.model.transformer import MathFormer, MathFormerConfig
from src.vis.server import start_server, update_state, get_latest_command, reset_state

# Hyperparameters
BATCH_SIZE = 64
MAX_ITERS = 5000
LEARNING_RATE = 3e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.backends.mps.is_available():
    DEVICE = 'mps'

class Trainer:
    def __init__(self):
        self.generator = MathGenerator()
        self.tokenizer = CharTokenizer()
        self.config = MathFormerConfig(
            vocab_size=self.tokenizer.vocab_size,
            block_size=128,
            n_layer=4,
            n_head=4,
            d_model=128
        )
        self.model = None
        self.optimizer = None
        self.iter_num = 0
        self.running = False
        self.difficulty = 1
        
        self.init_model()

    def init_model(self):
        print(f"Initializing model with config: {self.config}")
        self.model = MathFormer(self.config).to(DEVICE)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=LEARNING_RATE)
        self.iter_num = 0
        print("Model initialized.")

    def update_config(self, new_config):
        # Update config based on dict
        if 'n_layer' in new_config: self.config.n_layer = int(new_config['n_layer'])
        if 'n_head' in new_config: self.config.n_head = int(new_config['n_head'])
        if 'd_model' in new_config: self.config.d_model = int(new_config['d_model'])
        if 'difficulty' in new_config: self.difficulty = int(new_config['difficulty'])
        # Re-init model
        self.init_model()

    def step(self):
        # Generate batch
        batch_strs = self.generator.generate_batch(BATCH_SIZE, difficulty=self.difficulty)
        
        # Tokenize
        max_len = self.config.block_size
        X_batch = []
        Y_batch = []
        
        for s in batch_strs:
            ids = self.tokenizer.encode(s)
            if len(ids) > max_len:
                ids = ids[:max_len]
            pad_len = max_len - len(ids)
            padded_ids = ids + [self.tokenizer.pad_token_id] * pad_len
            X_batch.append(padded_ids[:-1])
            Y_batch.append(padded_ids[1:])
            
        X = torch.tensor(X_batch, dtype=torch.long).to(DEVICE)
        Y = torch.tensor(Y_batch, dtype=torch.long).to(DEVICE)
        
        # Forward
        logits, loss = self.model(X, Y)
        
        # Backward
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        
        loss_val = loss.item()
        
        # Visualization Update
        if self.iter_num % 10 == 0:
            print(f"Iter {self.iter_num}: Loss {loss_val:.4f}, Difficulty {self.difficulty}")
            with torch.no_grad():
                all_attn_weights = {}
                raw_attn = self.model.get_attention_weights()
                for i, layer_attn in enumerate(raw_attn):
                    num_heads = layer_attn.shape[1]
                    for h in range(num_heads):
                        all_attn_weights[f"L{i}.H{h}"] = layer_attn[0, h, :, :].cpu().numpy().tolist()
                
                all_weights = self.model.get_all_weights()
                input_text = self.tokenizer.decode(X[0].tolist())
                
                # use text before '=' to generate sample output
                output_prompt = input_text.split('=')[0] + '='
                with torch.no_grad():
                    # try infer for output text
                    x = torch.tensor([self.tokenizer.encode(output_prompt)], dtype=torch.long).to(DEVICE)
                    # Generate up to 20 new tokens
                    for _ in range(20):
                        logits, _ = self.model(x)
                        # Get next token from the last position
                        next_token = logits[0, -1, :].argmax()
                        if next_token.item() == self.tokenizer.pad_token_id:
                            break
                        x = torch.cat((x, next_token.unsqueeze(0).unsqueeze(0)), dim=1)
                    output_text = self.tokenizer.decode(x[0].tolist())
                
                embeddings = self.model.get_embeddings()
                vis_data = {
                    "layer_weights": all_weights,
                    "attention": all_attn_weights,
                    "embeddings": {
                        "token": embeddings["token"][0].numpy().tolist() if embeddings["token"] is not None else [],
                        "position": embeddings["position"][0].numpy().tolist() if embeddings["position"] is not None else []
                    },
                    "data_sample": {
                        "input": input_text,
                        "output": output_text
                    }
                }
                update_state(vis_data, self.iter_num, loss_val)
                
        if self.iter_num % 100 == 0:
            self.model.eval()
            with torch.no_grad():
                prompt = "10+10="
                x = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long).to(DEVICE)
                for _ in range(10):
                    logits, _ = self.model(x)
                    next_token = logits[0, -1, :].argmax()
                    x = torch.cat((x, next_token.unsqueeze(0).unsqueeze(0)), dim=1)
                    if next_token.item() == self.tokenizer.pad_token_id:
                        break
                decoded = self.tokenizer.decode(x[0].tolist())
                print(f"Sample: {decoded}")
            self.model.train()
            
        self.iter_num += 1

def main_loop():
    trainer = Trainer()
    trainer.running = True
    
    while True:
        # Check for commands
        cmd = get_latest_command()
        if cmd:
            print(f"DEBUG: Trainer received command: {cmd}")
            if cmd['action'] == 'restart':
                print("DEBUG: Executing restart...")
                trainer.update_config(cmd['config'])
                reset_state()
            elif cmd['action'] == 'update_difficulty':
                print(f"DEBUG: Updating difficulty to {cmd['config']['difficulty']}")
                trainer.difficulty = int(cmd['config']['difficulty'])
        
        if trainer.running:
            try:
                trainer.step()
            except Exception as e:
                print(f"Error in training step: {e}")
                time.sleep(1)
        
        # Small sleep to prevent 100% CPU if not training (though here we always train)
        # time.sleep(0.001)

if __name__ == "__main__":
    # Start Server in background thread
    server_thread = threading.Thread(target=start_server, kwargs={'port': 8000}, daemon=True)
    server_thread.start()
    
    print("Server started at http://localhost:8000")
    
    try:
        main_loop()
    except KeyboardInterrupt:
        print("Training stopped.")
