import torch
from transformers import LlamaForCausalLM, LlamaConfig, LlamaTokenizer
import argparse

parser = argparse.ArgumentParser(description="Convert the checkpoint to LlamaForCausalLM compatible format.")
parser.add_argument('ckpt', help='checkpoint path to convert')
parser.add_argument('tok', help='tokenizer path')
parser.add_argument('out', help='output path for the converted model')
args = parser.parse_args()

def rename_key(key):
    if key == 'tok_embeddings.weight':
        return 'model.embed_tokens.weight'
    elif key == 'norm.weight':
        return 'model.norm.weight'
    elif key == 'output.weight':
        return 'lm_head.weight'
    elif key.startswith('layers.'):
        layer_num = key.split('.')[1]
        sub_key = '.'.join(key.split('.')[2:])
        
        if sub_key == 'attention.wq.weight':
            return f'model.layers.{layer_num}.self_attn.q_proj.weight'
        elif sub_key == 'attention.wk.weight':
            return f'model.layers.{layer_num}.self_attn.k_proj.weight'
        elif sub_key == 'attention.wv.weight':
            return f'model.layers.{layer_num}.self_attn.v_proj.weight'
        elif sub_key == 'attention.wo.weight':
            return f'model.layers.{layer_num}.self_attn.o_proj.weight'
        elif sub_key == 'feed_forward.w1.weight':
            return f'model.layers.{layer_num}.mlp.gate_proj.weight'
        elif sub_key == 'feed_forward.w3.weight':
            return f'model.layers.{layer_num}.mlp.up_proj.weight'
        elif sub_key == 'feed_forward.w2.weight':
            return f'model.layers.{layer_num}.mlp.down_proj.weight'
        elif sub_key == 'attention_norm.weight':
            return f'model.layers.{layer_num}.input_layernorm.weight'
        elif sub_key == 'ffn_norm.weight':
            return f'model.layers.{layer_num}.post_attention_layernorm.weight'
    else:
        return None

# Load the state dict
orig_state_dict = torch.load(args.ckpt, weights_only=True)["model"]

# Create a new state dict with renamed keys
new_state_dict = {}
for old_key in orig_state_dict.keys():
    new_key = rename_key(old_key)
    new_state_dict[new_key] = orig_state_dict[old_key]
    
# Define model configuration
config = LlamaConfig(
    hidden_size=512,
    intermediate_size=1536,
    max_position_embeddings=128,
    num_attention_heads=8,
    num_hidden_layers=16,
    vocab_size=32000,
    rms_norm_eps=1e-5,
    initializer_range=0.02,
    use_cache=True,
    bos_token_id=1,
    eos_token_id=2,
    hidden_act="silu",
    torch_dtype="float16"
)

# Initialize the model
model = LlamaForCausalLM(config)

# Load the new state dict into the model
missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

# Check for any missing or unexpected keys
if missing_keys:
    print(f"Missing keys: {missing_keys}")
if unexpected_keys:
    print(f"Unexpected keys: {unexpected_keys}")


tokenizer = LlamaTokenizer.from_pretrained(args.tok)

model.save_pretrained(args.out)

tokenizer.save_pretrained(args.out)