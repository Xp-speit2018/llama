from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers.models.llama.modeling_llama import LlamaSdpaAttention, apply_rotary_pos_emb
import torch
import numpy as np
import types

def read_fvecs(filename):
    with open(filename, 'rb') as f:
        vecs = []
        while True:
            data = f.read(4)
            if len(data) < 4:
                break
            d = int.from_bytes(data, 'little')
            vec = np.frombuffer(f.read(d * 4), dtype=np.float32)
            vecs.append(vec)
        return np.array(vecs)

def write_fvecs(filename, vecs, mode='ab'):
    with open(filename, mode) as f:
        for vec in vecs:
            d = len(vec)
            f.write(np.int32(d).tobytes())
            f.write(vec.astype(np.float32).tobytes())

def save_forward(
    self: LlamaSdpaAttention,
    hidden_states,
    attention_mask = None,
    position_ids = None,
    past_key_value = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position = None,
):

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # save key to files, per layer per head to .fvecs filess
    for b in range(bsz):
        for h in range(self.num_heads):
            key = key_states[b, h].view(q_len, self.head_dim).cpu().detach().numpy()
            write_fvecs(f'../llama_key/key_{self.layer_idx}_{h}.fvecs', key, 'ab')

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        # cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)
    
    causal_mask = None

    # In case we are not compiling, we may set `causal_mask` to None, which is required to dispatch to SDPA's Flash Attention 2 backend, rather
    # relying on the `is_causal` argument.
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=causal_mask is None and q_len > 1,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value

def main():
    model = LlamaForCausalLM.from_pretrained('../llama-2-7b-hf')
    tokenizer = LlamaTokenizer.from_pretrained('../llama-2-7b-hf')

    # replace the forward function with the custom one
    for layer in model.model.layers:
        layer.self_attn.forward = types.MethodType(save_forward, layer.self_attn)

    import sys
    # add parent path
    sys.path.append('../')

    from perplexity import perplexity

    device = 'cuda:0'
    root = '../'
    dataset = ''

    stride = model.config.max_position_embeddings # 4096
    ppl_baseline = perplexity(model, tokenizer, dataset, device, verbose=True, stride=stride, root=root)
    print(ppl_baseline)

    # move ../llama_key/*.fvecs to ../llama_key/PTB/*.fvecs

    import os
    import shutil
    import glob

    # 创建目标目录
    os.makedirs(f'../llama_key/{dataset}', exist_ok=True)

    # 获取所有符合通配符的文件
    files = glob.glob('../llama_key/*.fvecs')

    # 移动每个文件到目标目录
    for file in files:
        shutil.move(file, f'../llama_key/{dataset}/')

if __name__ == "__main__":
    main()
