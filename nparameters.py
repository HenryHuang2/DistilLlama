from config import ModelArgs
def calculate_nparameters(args: ModelArgs, vocab_size: int) -> int:
    n_parameters = 0
    n_parameters_embedding = vocab_size * args.dim
    if args.n_kv_heads is None:
        n_parameters_attention = args.dim * args.dim * 4 # wq, wk, wv, wo
    else:
        n_parameters_attention = args.dim * (args.dim / args.n_heads * args.n_kv_heads) + 2 * args.dim * args.dim 
        # wk and wv: dim * (head_dim * n_kv_heads)
        # wq and wo: dim * dim
    n_parameters_ffn = 3 * (args.dim * (args.dim * 4) * 2/3) 
    # hidden dimension = dim * 4 * 2/3, w1, w3 has dimension of dim * hidden_dim, w2 has dimension of hidden_dim * dim
    
    n_parameters_rms = args.dim * 2
    
    n_parameters_out = args.dim * vocab_size
    
    n_parameters += n_parameters_embedding
    n_parameters += (n_parameters_attention + n_parameters_ffn + n_parameters_rms * 2) * args.n_layers
    n_parameters += n_parameters_rms
    n_parameters += n_parameters_out
    
    return int(n_parameters)


if __name__ == '__main__':
    args = ModelArgs()
    print(calculate_nparameters(args, 32000))