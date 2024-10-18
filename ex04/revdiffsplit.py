
%%file revdiffgpt_split_v2.py 

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import time
from tiktoken import encoding_for_model
import os
import datetime
import unicodedata
print("Starting script execution...")

def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)

def generate_causal_mask(T, device):
    causal_mask = torch.triu(torch.ones((T, T), device=device), diagonal=1).bool()
    return causal_mask
    
def apply_rotary_embedding(x, seq_len):
    #print(f"Applying rotary embedding. Input shape: {x.shape}, seq_len: {seq_len}")
    dim = x.shape[-1]
    dtype = x.dtype
    theta = 10000.0
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    freqs = torch.outer(torch.arange(seq_len), freqs).to(x.device)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    
    if dtype not in [torch.float32, torch.float64]:
        x_temp = x.float()
    else:
        x_temp = x
    
    x_temp = x_temp.view(*x_temp.shape[:-1], x_temp.shape[-1] // 2, 2)
    x_temp = torch.view_as_complex(x_temp)
    x_temp = x_temp * freqs_cis
    x_rot = torch.view_as_real(x_temp).view(*x.shape[:-1], -1)
    
    if x.dtype != x_rot.dtype:
        x_rot = x_rot.to(dtype)
    
    #print(f"Rotary embedding applied. Output shape: {x_rot.shape}")
    return x_rot

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        #print(f"Initializing CausalSelfAttention with config: {config}")
        self.config = config
        self.n_embd = config.n_embd // 2  ## for reversible dual path
        self.n_head = config.n_head
        self.head_dim =  self.n_embd // self.n_head
                
        # Linear projections for Q, K, V
        self.c_attn = nn.Linear(self.n_embd, 3 * (self.n_embd))
        self.c_proj = nn.Linear(self.n_embd, self.n_embd)
        
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.scaling = self.head_dim ** -0.5

        # Lambda for differential attention
        self.lambda_init = lambda_init_fn(config.n_layer)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        
        # LayerNorm inside the attention mechanism
        #self.ln_attn = nn.LayerNorm(self.n_embd )

        # GroupNorm for normalizing across heads (1 group per head)
        self.gn_attn = nn.GroupNorm(1, self.head_dim, affine=False)  # GroupNorm with n_head groups

    
        
    def forward(self, x, attn_mask=None):
        # Apply half-size LayerNorm inside attention
        #x = self.ln_attn(x)  # Apply half-size LayerNorm
        
        #print(f"CausalSelfAttention forward pass. Input shape: {x.shape}")
        B, T, C = x.size()
        
        # Apply linear projection to get Q, K, V
        qkv = self.c_attn(x) # Shape: [B, T, 3 * n_embd]
        q, k, v = qkv.split(self.n_embd, dim=2)  # Split into Q, K, V
        
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # [B, n_head, T, head_dim]        
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # [B, n_head, T, head_dim]        
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # [B, n_head, T, head_dim]
        
        # Split Q and K into Q1, Q2 and K1, K2
        q1, q2 = torch.chunk(q, 2, dim=-1)  # Split into Q1 and Q2
        k1, k2 = torch.chunk(k, 2, dim=-1)  # Split into K1 and K2

        # Apply RoPE (Rotary Positional Embedding) to Q1, Q2, K1, K2
        q1 = apply_rotary_embedding(q1, T)
        q2 = apply_rotary_embedding(q2, T)
        k1 = apply_rotary_embedding(k1, T)        
        k2 = apply_rotary_embedding(k2, T)

        # Scale queries
        q1 = q1 * self.scaling
        q2 = q2 * self.scaling
        

        # Compute attention scores
        attn_scores_1 = torch.matmul(q1, k1.transpose(-1, -2))  # [B, n_head, T, T]
        attn_scores_2 = torch.matmul(q2, k2.transpose(-1, -2))  # [B, n_head, T, T]

        # Apply causal mask (upper triangular mask)
        if attn_mask is None:
            causal_mask = torch.triu(torch.ones((T, T), device=x.device), diagonal=1).bool()
            attn_scores_1 = attn_scores_1.masked_fill(causal_mask, float('-inf'))  # Mask for Q1K1
            attn_scores_2 = attn_scores_2.masked_fill(causal_mask, float('-inf'))  # Mask for Q2K2
        else:
            attn_scores_1 += attn_mask
            attn_scores_2 += attn_mask            

        # Softmax for both sets of scores
        attn_weights_1 = F.softmax(attn_scores_1, dim=-1)  # [B, n_head, T, T]
        attn_weights_2 = F.softmax(attn_scores_2, dim=-1)  # [B, n_head, T, T]

        # Compute lambda values
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1)).type_as(q1)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1)).type_as(q2)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        # Differential attention: Subtract weighted attention scores
        attn_weights = attn_weights_1 - lambda_full * attn_weights_2

        # Apply attention weights to values (V)
        attn_output = torch.matmul(attn_weights, v)  # [B, n_head, T, head_dim]

 
        # Reshape to apply GroupNorm on each attention head independently
        attn_output = attn_output.view(B * self.n_head, T, self.head_dim)  # [B * n_head, T, head_dim]

        # Apply GroupNorm within each head
        attn_output = self.gn_attn(attn_output)  # [B * n_head, T, head_dim]

        # Reshape back to [B, T, n_embd]
        attn_output = attn_output.view(B, self.n_head, T, self.head_dim).transpose(1, 2).contiguous().view(B, T, self.n_embd)

        # Final linear projection 
        output = self.c_proj(attn_output)  # [B, T, n_embd]
        
        return output


class ChunkedFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        #print(f"Initializing ChunkedFFN with config: {config}")
        self.c_fc = nn.Linear(config.n_embd // 2, int(2 * (config.n_embd // 2)))
        self.c_proj = nn.Linear(int(2 * (config.n_embd // 2)), config.n_embd // 2)
        self.chunk_size = 16  # Default chunk size, can be adjusted
        self.c_proj.NANOGPT_SCALE_INIT = 1
        
        self.ln_ffn = nn.LayerNorm(config.n_embd // 2)  # Half-size LayerNorm for FFN        
    

    def forward(self, x):
        # x = self.ln_ffn(x)  # Apply half-size LayerNorm        
        #print(f"ChunkedFFN forward pass. Input shape: {x.shape}")
        B, T, C = x.size()
        chunks = x.chunk(self.chunk_size, dim=1)
        outputs = []
        for chunk in chunks:
            chunk = F.gelu(self.c_fc(chunk))
            chunk = self.c_proj(chunk)
            outputs.append(chunk)
        output = torch.cat(outputs, dim=1)
        #print(f"ChunkedFFN output shape: {output.shape}")
        output = self.ln_ffn(output)  # Apply half-size LayerNorm        
        return output


class ReversibleLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.ffn = ChunkedFFN(config)

    def forward(self, x1, x2, attn_mask=None):
        # Reversible core computations
        y1 = x1 + self.attn(x2, attn_mask=attn_mask)
        y2 = x2 + self.ffn(y1)
        return y1, y2

    def backward_pass(self, y1, y2, attn_mask=None):
        # Reversible core computations during the backward pass
        x2 = y2 - self.ffn(y1)
        x1 = y1 - self.attn(x2, attn_mask=attn_mask)
        return x1, x2

class ReversibleGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = encoding_for_model(config.tokenizer_name)
        self.vocab_size = self.tokenizer.n_vocab
        self.wte = nn.Embedding(self.vocab_size, config.n_embd)
        self.layers = nn.ModuleList([ReversibleLayer(config) for _ in range(config.n_layer)])
        self.ln_f_half = nn.LayerNorm(config.n_embd//2)
        self.ln_f_full1 = nn.LayerNorm(config.n_embd)
        self.ln_f_full2 = nn.LayerNorm(config.n_embd)        
        self.lm_head = nn.Linear(config.n_embd, self.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, attn_mask=None ):
        B, T = input_ids.size()

        # Token embedding
        x = self.wte(input_ids)
        

        # Apply LayerNorm before entering the reversible blocks

         
        # Split input embedding into x1 and x2
        x1, x2 = torch.chunk(x, 2, dim=-1)

        # Pass through reversible layers
        for layer in self.layers:
          
            x1, x2 = layer(x1, x2, attn_mask=attn_mask)
        
        # Concatenate the two halves
        x = torch.cat([x1, x2], dim=-1)

        # Apply LayerNorm after reversible layers
        x = self.ln_f_full2(x)
        
        # Compute logits
        logits = self.lm_head(x)

        loss = None
 
        # Shift targets by one position to predict the next token
        targets = input_ids[:, 1:].contiguous()  # Shape: [batch_size, T]
        logits = logits[:, :-1, :].contiguous()  # Shape: [batch_size, T, vocab_size]
    
        # Calculate loss (flatten the logits and targets for batch processing)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

        
class TimedParquetDataset(Dataset):
    def __init__(self, file_path, tokenizer_name='gpt-4o', columns=None):
        print(f"Initializing TimedParquetDataset with file_path: {file_path}, tokenizer_name: {tokenizer_name}")
        start_time = time.time()
        
        self.df = pd.read_parquet(file_path, columns=columns)
        
        end_time = time.time()
        print(f"Parquet file loaded. Total load time: {end_time - start_time:.6f} seconds")
        print(f"Total rows loaded: {len(self.df)}")
        
        self.tokenizer = encoding_for_model(tokenizer_name)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx].to_dict()
        text = " ".join(str(value) for value in row.values())
        tokenized_text = self.tokenizer.encode(text)
        tokenized_tensor = torch.tensor(tokenized_text, dtype=torch.long)

        #print(f"Retrieved item {idx}. Tokenized tensor shape: {tokenized_tensor.shape}")
        return {
            "input_ids": tokenized_tensor,
            "row_idx": idx,
        }

class CustomCollate:
    def __init__(self, ctx_len=8192, batch_size=2, add_token=" ", add_special_token=None, add_bos_token=None, metadata=False):
        self.ctx_len = ctx_len
        self.batch_size = batch_size
        self.add_token = add_token
        self.add_special_token = add_special_token
        self.add_bos_token = add_bos_token
        self.metadata = metadata

        tokenizer = encoding_for_model('gpt-4o')
        self.bos_token_id = tokenizer.encode("<|endoftext|>", allowed_special={'<|endoftext|>'})[0] if add_bos_token else None
        self.merge_token_id = tokenizer.encode(self.add_token)[0] if self.add_token is not None else None
        self.special_token_id = tokenizer.encode(self.add_special_token)[0] if self.add_special_token is not None else None

    def __call__(self, batch):
        merged_sequences = []
        batch_meta_data = []
        current_buffer = []
        current_token_count = 0
        current_meta_data = {'row_idx': []}

        all_input_ids = [item['input_ids'] for item in batch]
        all_row_idxs = [item['row_idx'] for item in batch]

        for input_ids, row_idx in zip(all_input_ids, all_row_idxs):
            current_meta_data['row_idx'].append(row_idx)

            if self.bos_token_id is not None and len(current_buffer) == 0:
                current_buffer.append(self.bos_token_id)

            current_buffer.extend(input_ids.tolist())
            if self.merge_token_id is not None:
                current_buffer.append(self.merge_token_id)

            current_token_count += len(input_ids)

            if current_token_count >= self.ctx_len:
                # Ensure buffer is exactly ctx_len, either by truncating or padding
                if len(current_buffer) > self.ctx_len:
                    current_buffer = current_buffer[:self.ctx_len]
                elif len(current_buffer) < self.ctx_len:
                    current_buffer.extend([0] * (self.ctx_len - len(current_buffer)))

                merged_sequences.append(torch.tensor(current_buffer, dtype=torch.long))

                batch_meta_data.append({
                    'row_idx': current_meta_data['row_idx'],
                    'total_token_count': current_token_count,
                })

                current_buffer = []
                current_token_count = 0
                current_meta_data = {'row_idx': []}

                if len(merged_sequences) == self.batch_size:
                    break

        # Handle remaining tokens in the buffer
        if len(current_buffer) > 0 and len(merged_sequences) < self.batch_size:
            # Ensure the final buffer is exactly ctx_len
            if len(current_buffer) < self.ctx_len:
                current_buffer.extend([0] * (self.ctx_len - len(current_buffer)))
            else:
                current_buffer = current_buffer[:self.ctx_len]

            merged_sequences.append(torch.tensor(current_buffer, dtype=torch.long))
            batch_meta_data.append({
                'row_idx': current_meta_data['row_idx'],
                'total_token_count': current_token_count,
            })

        # Ensure we have a full batch by padding with empty sequences if necessary
        while len(merged_sequences) < self.batch_size:
            merged_sequences.append(torch.zeros(self.ctx_len, dtype=torch.long))

        # Stack the sequences to form the batch tensor
        batch_tensor = torch.stack(merged_sequences)
        output = {'input_ids': batch_tensor}
        if self.metadata:
            output['meta_data'] = batch_meta_data

        return output


def calculate_pnorm_gnorm(model):
    # Calculate the parameter norm (PNorm)
    param_norm = torch.norm(torch.stack([torch.norm(p, 2) for p in model.parameters() if p.requires_grad]))

    # Calculate the gradient norm (GNorm), which should be zero before training
    grad_norm = torch.norm(torch.stack([torch.norm(p.grad, 2) for p in model.parameters() if p.grad is not None]), p=2) if any(p.grad is not None for p in model.parameters()) else 0.0

    return param_norm.item(), grad_norm if grad_norm == 0 else grad_norm.item()


def train(config):


    # Add this before the training loop to create the log file with a timestamp
    log_filename = f"log_split_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    print(f"Starting training with config: {vars(config)}")
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    seed_value = 1337 
    print(f"Random seed: {seed_value}")    
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.set_float32_matmul_precision('medium') # : highest : fp32 high tf32,   medium : bf16
        
    print("Configuring model")
    model = ReversibleGPT(config).to(device)
    print(model)


    T = config.ctx_len  # assuming ctx_len is the sequence length
    causal_mask = generate_causal_mask(T, config.device)
    
    print("initialize parameters") 
    # Apply weight initialization
    model.apply(model._init_weights)
    

    # Calculate and print PNorm and GNorm before training
    pnorm, gnorm = calculate_pnorm_gnorm(model)
    print(f"Initial PNorm: {pnorm:.6f}, Initial GNorm: {gnorm:.6f}")
    

    print("Configuring optimizer")
    optimizer = optim.AdamW(model.parameters(), lr=config.lr ) #, weight_decay=0.01)

    
    print("Configuring dataset")
    dataset = TimedParquetDataset(config.file_path, tokenizer_name=config.tokenizer_name)
    collate_fn = CustomCollate(
        ctx_len=config.ctx_len,
        batch_size=config.batch_size,
        add_token=" ",
        add_bos_token=False,
        metadata=True
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    os.makedirs(config.checkpoint_path, exist_ok=True)
    print(f"Checkpoint directory created/verified: {config.checkpoint_path}")
    tokenizer = encoding_for_model(config.tokenizer_name)
    
    print("Initializing model training")
    model.train()
    total_tokens = 0
    total_tokens_with_truncated = 0
    start_time = time.perf_counter()

    print("Starting iterations")
    for iteration in range(config.num_iterations):
        for idx, batch in enumerate(dataloader):
            inputs = batch['input_ids'].to(device)

            meta_data = batch['meta_data'][0]

            optimizer.zero_grad()

            # Forward pass
            forward_start = time.perf_counter()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=config.use_amp):
                logits, loss = model(inputs, attn_mask=causal_mask )
            forward_end = time.perf_counter()


            # Decode input, ground truth, and prediction
            input_text = tokenizer.decode(inputs[0].tolist()).replace("\n"," ").replace("\r", " ")
            # Get predicted tokens for both paths
            pred_ids = torch.argmax(logits[0], dim=-1)            
            pred_text = tokenizer.decode(pred_ids.tolist()).replace("\n"," ").replace("\r", " ")      

            
            # Backward pass
            backward_start = time.perf_counter()
            if loss is not None:
                loss.backward()
            backward_end = time.perf_counter()

            # Calculate norms
            grad_norm = torch.norm(torch.stack([p.grad.norm(2) for p in model.parameters() if p.grad is not None]))
            param_norm = torch.norm(torch.stack([p.norm(2) for p in model.parameters()]))

            # Debug prints for backward pass
            #print(f"Backward pass - Grad norm: {grad_norm.item()}")
            #print(f"First layer gradient: {model.layers[0].attn.c_attn.weight.grad.norm().item() if model.layers[0].attn.c_attn.weight.grad is not None else 'None'}")

            # Optimizer step
            optimizer_start = time.perf_counter()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            optimizer_end = time.perf_counter()        
            
            batch_tokens = inputs.numel()
            total_tokens += batch_tokens
            total_tokens_with_truncated += meta_data['total_token_count']

            forward_time = (forward_end - forward_start) * 1000
            backward_time = (backward_end - backward_start) * 1000
            optimizer_time = (optimizer_end - optimizer_start) * 1000

            debug_info_default = f"ep {iteration :>2d} | idx {idx:>6d}"

            if config.debug_level >= 1:
                debug_info_default += ( # f" | Rows: {meta_data['row_idx'][0]:>8d}-{meta_data['row_idx'][-1]:>8d}"
                               f" | Tokens: {batch_tokens:>6d} | Total: {total_tokens/1e6:>8.4f}M"
                               #f" | WithTrunc: {total_tokens_with_truncated/1e6:>8.4f}M"
                               f" | Loss: {loss.item() if loss is not None else 0:>12.8f}"
                               f" | PNorm: {param_norm:>10.4f} | GNorm: {grad_norm:>10.4f}"
                               f" | FwdT: {forward_time:>3.1f}ms | BwdT: {backward_time:>3.1f}ms | OptT: {optimizer_time:>3.1f}ms"               
                )

                fixed_width = 80  # Set the fixed width for the text fields
 
                # Helper function to calculate the display width of Unicode text
                def get_display_width(text):
                    return sum(2 if unicodedata.east_asian_width(char) in 'WF' else 1 for char in text)
                
                # Helper function to adjust text to fixed width with Unicode awareness
                def adjust_to_fixed_width(text, width):
                    display_width = get_display_width(text)
                    
                    if display_width > width:
                        # Truncate text to fit within the width
                        truncated_text = ''
                        current_width = 0
                        for char in text:
                            char_width = 2 if unicodedata.east_asian_width(char) in 'WF' else 1
                            if current_width + char_width > width - 3:  # Reserve space for "..."
                                break
                            truncated_text += char
                            current_width += char_width
                        return truncated_text + '...'
                    else:
                        # Pad text with spaces to ensure fixed width
                        padding = width - display_width
                        return text + ' ' * padding
                
                # Apply the helper function to each text field
                input_fixed      = adjust_to_fixed_width(input_text, fixed_width)
                pred_text_fixed  = adjust_to_fixed_width(pred_text, fixed_width)

                debug_info_default = debug_info_default + (
                                f" ||| input_text : {input_fixed}"
                                f" | | pred_text  : {pred_text_fixed}"            
                )                

                print(debug_info_default)
            
                debug_info_save =debug_info_default +  (
                    f"\nep {iteration :>2d} | idx {idx:>6d} ||  input      : {input_text}"
                    f"\nep {iteration :>2d} | idx {idx:>6d} ||  pred_text  : {pred_text }"
           
                    
                )
                # Save to log file
                with open(log_filename, 'a') as log_file:
                    log_file.write(debug_info_save + '\n')  # Append each debug_info with a newline
                    
            
            if idx % config.checkpoint_interval == 0 and idx == 0:
                checkpoint_path = os.path.join(config.checkpoint_path, f'checkpoint_{iteration}.pt')
                print(f"Saving checkpoint to {checkpoint_path}")
                torch.save({
                    'iteration': iteration,
                    'idx': idx,
                    'model_state_dict': model.state_dict(),
                    #'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, checkpoint_path)

        print(f"Completed iteration {iteration}")

    print("Training completed")
    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f"Training completed in {total_time:>4.2f} seconds")
    print(f"Total tokens processed (seen): {total_tokens / 1e6:>8.2f}M")
    print(f"Total tokens processed (with truncated): {total_tokens_with_truncated / 1e6:>8.2f}M")
    

class Config:
    def __init__(self):
        self.file_path = "/mnt/e/jupyter/00_shared_dataset/syntheric/random_sentences.parquet"
        self.checkpoint_path = "./checkpoints/"
        self.device = "cuda:0"
        self.tokenizer_name = "gpt-4o"
        self.ctx_len = 1024
        self.n_layer = 4
        self.n_embd = 512
        self.n_head = 4
        self.use_amp = True
        self.batch_size = 6
        self.lr = 1e-3
        self.num_iterations = 10
        self.checkpoint_interval = 10000
        self.log_interval = 1
        self.grad_clip = 1
        self.debug_level = 1  # Add this line
        

# Main execution
if __name__ == "__main__":
    print("Starting main execution")
    
    config = Config()

    # You can modify config parameters here if needed
    # config.num_iterations = 200
    # config.batch_size = 4
    
    train(config)
    print("Main execution completed")
