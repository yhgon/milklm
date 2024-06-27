# exp 01 model card

this is experiment 01 for small dataset
- it use 20366MiB with 50 batch
- 32L x [ ( 1024 dim, 16 heads, no GQA, non-trainable projection 1/10 proj)  + 0.5 scale ffn ] : 372MParam ~ mimic 3B param model
- dataset : 2024 fineweb-edu only 4.4GB(2EA 2.2GB) ,  1_577_715_712 tokens   1.5GTokens 

-  50batch x 64Tokens clm
- 
```
ModelConfig(embedding_size=1024, num_layers=32, num_heads=16, ffn_scale_ratio=0.5, max_context_length=8192, vocab_size=200019, debug=False, proj_ratio_min=2, proj_ratio=10, num_keep_boundary_chunk=2)
DataConfig(batch_size=50, block_size=64, buffer_size=10, tokenizer_name='gpt-4o', dataset_dir='datasets/finewebedu/CC-MAIN-2024-10', truncate_limit=1000000000, shuffle_tokens=False, mask_random=False)
TrainConfig(learning_rate=0.001, dtype='bfloat16', optimizer_options=None, exp_name='experiment', exp_num=6, save_interval_epoch=10, save_interval_iter=1000, print_interval_iteration=100, eval_interval_iteration=250, num_epochs=200, max_total_iters=10000000, num_warmup_steps=4000, gradient_checkpointing=True, value_clip_grad_norm=0.5, lr_step=1000, print_token_len=64)
model configure  3.9sec
configure optimizer during  0.4sec
372.90496 M Params
Transformer(
  (transformer): ModuleDict(
    (wte): Embedding(200019, 1024)
    (h): ModuleList(
      (0-31): 32 x Block(
        (ln_1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        (attn): CausalSelfAttention(
          (c_attn): Linear(in_features=1024, out_features=3072, bias=True)
          (c_proj): Linear(in_features=1024, out_features=1024, bias=True)
        )
        (ln_2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        (mlp): MLP(
          (c_fc): Linear(in_features=1024, out_features=512, bias=True)
          (gelu): GELU(approximate='none')
          (c_proj): Linear(in_features=512, out_features=1024, bias=True)
        )
      )
    )
    (ln_f): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=1024, out_features=200019, bias=False)
)
```
