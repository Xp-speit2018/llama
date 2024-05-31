from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, rotate_half, repeat_kv, LlamaDecoderLayer, LlamaMLP, LlamaRMSNorm, LlamaModel, LlamaSdpaAttention, LlamaPreTrainedModel
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Optional, Tuple, Union
import torch
import math

import faiss
from faiss import IndexPQ, IndexFlatIP
from faiss import read_index

from pyinstrument import Profiler
from memory_profiler import memory_usage
from tqdm import tqdm

# profiler = Profiler()

from pynvml.smi import nvidia_smi
nvsmi = nvidia_smi.getInstance()

GPU_MAX_SELECTION_K = 2048

def gpu_profile(func):
    def wrapper(*args, **kwargs):
        # mem_usage_before = memory_usage(-1,interval=0.1, timeout=1)
        # start = time.time()
        print("before", nvsmi.DeviceQuery('memory.free, memory.total'))
        result = func(*args, **kwargs)
        print("after", nvsmi.DeviceQuery('memory.free, memory.total'))
        # stop = time.time()
        # mem_usage_after = memory_usage(-1,interval=0.1, timeout=1)
        # plt.figure(figsize=(10,5))plt.plot(mem_usage_before,'-b', label='Before')plt.plot(mem_usage_after,
        # plt.ylabel('Memory usage(MB)')plt.xlabel('Time(s)')plt.legend(loc='best')
        # plt.show()
        # print(f'Time elapsed: {stop - start}')
        # print(f"mem_usage_before: {mem_usage_before}")
        # print(f"mem_usage_after: {mem_usage_after}")
        return result
    return wrapper

def profile(func):
    def wrapper(*args, **kwargs):
        mem_usage_before = memory_usage(-1,interval=0.1, timeout=1)
        # start = time.time()
        result = func(*args, **kwargs)
        # stop = time.time()
        mem_usage_after = memory_usage(-1,interval=0.1, timeout=1)
        # plt.figure(figsize=(10,5))plt.plot(mem_usage_before,'-b', label='Before')plt.plot(mem_usage_after,
        # plt.ylabel('Memory usage(MB)')plt.xlabel('Time(s)')plt.legend(loc='best')
        # plt.show()
        # print(f'Time elapsed: {stop - start}')
        print(f"mem_usage_before: {mem_usage_before}")
        print(f"mem_usage_after: {mem_usage_after}")
        return result
    return wrapper

def apply_rotary_pos_emb_single(k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `torch.Tensor` comprising of the key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return k_embed

# @profile
def restore_index(layer_idx: int, head_idx: int, toGPU=False):
    """
    Restore a FAISS IndexPQ from disk.
    
    Args:
    layer_idx (int): The index of the layer in the model.
    head_idx (int): The index of the attention head within the layer.

    Returns:
    faiss.IndexPQ: The restored FAISS index.
    """
    index_filename = f"../llama_pqindex/PTB/key_{layer_idx}_{head_idx}.ivf" # TODO: use ivfpq with nlist=1 to move to GPU
    # index_filename = f"../pq_index/pq_{layer_idx}_{head_idx}.index"
    idx = read_index(index_filename)
    if toGPU is True:
        # move the index to gpu
        res = faiss.StandardGpuResources()
        res.noTempMemory() #TODO: install nightly build https://github.com/facebookresearch/faiss/issues/3259
        idx = faiss.index_cpu_to_gpu(res, 0, idx)
        # print(f"pq_{layer_idx}_{head_idx}.index moved to GPU.")

        # copy to ivf index
        
        
    return idx

class KeyStateTensorMocker:
    def __init__(self, key_states: Union[torch.Tensor, 'KeyStateTensorMocker'], layer_idx: int) -> None:
        self._cache = None
        self._shape = None
        # self._debug_cache = None
        
        self.layer_idx = layer_idx
        if key_states is not None:
            if isinstance(key_states, KeyStateTensorMocker):
                # This is used when from_legacy_cache is called
                self._cache = key_states._cache
                self._shape = key_states._shape
                # self._debug_cache = key_states._debug_cache
                return

            bsz, num_heads, seq_len, head_dim = key_states.shape
            self._cache = [IndexFlatIP(head_dim) for _ in range(num_heads)]
            # self._cache = [restore_index(layer_idx, i) for i in range(num_heads)] # TODO: support bsz>1
            self._shape = [bsz, num_heads, 0, head_dim]

            self.cat(key_states)          

    @property
    def shape(self) -> Optional[Tuple[int, int, int, int]]:
        # Return the shape if available
        return tuple(self._shape)
    
    def cat(self, key_states: torch.Tensor) -> None:
        '''
        Update or init the cache with key_states.
        '''
        bsz, num_heads, seq_len, head_dim = key_states.shape
        assert head_dim == self._shape[-1], "The head dimension of the key_states does not match the cache's head dimension"
        assert num_heads == self._shape[1], "The number of heads of the key_states does not match the cache's number of heads"

        for b in range(bsz):
            for i in range(num_heads):
                self._cache[i].add(key_states[b, i, :, :].cpu().numpy())

        self._shape[2] += seq_len

        # if self._debug_cache is None:
        #     self._debug_cache = key_states
        # else:
        #     self._debug_cache = torch.cat([self._debug_cache, key_states], dim=-2)
 
        pass
    
    def __getitem__(self, idx: int) -> IndexFlatIP:
        # print("idx", idx)
        return self._cache[idx]

    def reconstruct(self) -> torch.Tensor:
        '''
        Reconstruct the mocker to a torch.Tensor
        Inefficient, only for debugging
        '''
        bsz, num_heads, seq_len, head_dim = self._shape
        key_states = torch.zeros(bsz, num_heads, seq_len, head_dim, device='cuda:0')

        for b in range(bsz):
            for i in range(num_heads):
                key_states[b, i, :, :] = torch.tensor(self._cache[i].reconstruct_n(0, seq_len), device=key_states.device)
        
        return key_states

class DatabaseCache(DynamicCache):
    def __init__(self) -> None:
        self.key_cache : List[KeyStateTensorMocker] = [] # indexed by layer_idx
        self.value_cache : List[torch.Tensor] = []
        # self._debug_key_cache : List[torch.Tensor] = []
        self._seen_tokens = 0
    
    def reorder_cache(self, beam_idx: torch.LongTensor):
        raise NotImplementedError("Reordering the cache is not currently supported")

    def query(self, query_states, layer_idx, *, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, transition_matrix=None, cos=None, sin=None) -> torch.Tensor:
        '''
        Basically implements SDPA with cache
        '''
        bsz, num_heads, query_len, head_dim = query_states.shape
        seq_len = self._seen_tokens

        assert bsz == 1, "Batch size > 1 is not currently supported"
        
        # scale
        scaling_factor = 1 / math.sqrt(head_dim) if scale is None else scale
        # scaling_factor = 1

        # bias
        attn_bias = torch.zeros(query_len, seq_len, device=query_states.device, dtype=query_states.dtype)
        if is_causal:
            assert attn_mask is None, "is_causal and attn_mask cannot be used together"
            temp_mask = torch.ones(query_len, seq_len, device=query_states.device, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float('-inf'))
            attn_bias.to(query_states.dtype)
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill(attn_mask.logical_not(), float('-inf'))
            else:
                attn_bias += attn_mask

        # score
        if seq_len > 0:
            attn_score = torch.zeros(bsz, num_heads, query_len, seq_len, device=query_states.device)
            top_k = int(seq_len * 1)
            # if top_k > GPU_MAX_SELECTION_K:
            #     # red warning
            #     print(f"\033[91mWarning: top_k is {top_k}, which is larger than GPU_MAX_SELECTION_K {GPU_MAX_SELECTION_K}.\033[0m")
            #     top_k = GPU_MAX_SELECTION_K
            # top_k = seq_len
            for b in range(bsz): # TODO: parallelize this
                for h in range(num_heads):
                    
                    search = self.key_cache[layer_idx][h].search
                    # search = profile(self.key_cache[layer_idx][h].search)
                    D, I = search(query_states[b, h, :, :].cpu().numpy(), top_k) # TODO: specify k
                    # convert D to tensor
                    D = torch.tensor(D, device=query_states.device)
                    I = torch.tensor(I, device=query_states.device)
                    
                    attn_score[b, h, torch.arange(I.size(0)).unsqueeze(1), I] = D
        
        # print(f"layer {layer_idx} score done")

        # else:
        #     attn_score = query_states @ self._debug_key_cache[layer_idx].transpose(-1, -2)


        attn_score[torch.abs(attn_score) < 1e-5] = -1e10

        attn_score = attn_score * scaling_factor + attn_bias

        # softmax
        attn_score = torch.softmax(attn_score, dim=-1)

        # dropout
        attn_score = torch.dropout(attn_score, dropout_p, train=True)
        
        
        # restore v from k
        key_rot = self.key_cache[layer_idx].reconstruct()
        key_rerot = apply_rotary_pos_emb_single(key_rot, cos, -sin)
        del key_rot
        key_rerot_full_head = key_rerot.transpose(1, 2).contiguous().view(bsz, self._seen_tokens, num_heads*head_dim)
        del key_rerot # contiguous will copy the tensor, so we can delete the original tensor
        value_restored =(key_rerot_full_head @ transition_matrix).view(bsz, self._seen_tokens, num_heads, head_dim).transpose(1, 2)
        del key_rerot_full_head
        
        # weighted sum
        # return attn_score @ self.value_cache[layer_idx]
        return attn_score @ value_restored

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None) -> None:
        '''
        Broken change: returns None instead of updated states
        '''
        # key_states is shaped (bsz, num_heads, seq_len, head_dim)
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        bsz, num_heads, seq_len, head_dim = key_states.shape

        # initialize the cache if it doesn't exist
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(KeyStateTensorMocker(key_states, layer_idx))
            self.value_cache.append(value_states)

            if isinstance(key_states, KeyStateTensorMocker):
                key_states = key_states.reconstruct()
                # key_states = key_states._debug_cache
            # self._debug_key_cache.append(key_states)
        else:
            # update the cache
            self.key_cache[layer_idx].cat(key_states)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            # self._debug_key_cache[layer_idx] = torch.cat([self._debug_key_cache[layer_idx], key_states], dim=-2)
     
class LlamaForCausalLMDB(LlamaForCausalLM):
    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlamaModelDB(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        self.middleware = {}
        self.fwcall = 0

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        self.fwcall += 1
        print(f"fwcall: {self.fwcall}, key cache size(one layer): {past_key_values[0][0].shape if past_key_values is not None else None}")
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
class LlamaModelDB(LlamaModel):
    def __init__(self, config):
        super(LlamaModel, self).__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayerDB(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

        self.middleware = {}
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        '''
        Only difference is that we use DatabaseCache instead of DynamicCache
        '''

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_seen_tokens = 0
        if use_cache:  # kept for BC (cache positions)
            if not isinstance(past_key_values, StaticCache):
                past_key_values = DatabaseCache.from_legacy_cache(past_key_values)
                past_seen_tokens = past_key_values.get_seq_length()

        if cache_position is None:
            if isinstance(past_key_values, StaticCache):
                raise ValueError("cache_position is a required argument when using StaticCache.")
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_seen_tokens)

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
            )
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class LlamaDecoderLayerDB(LlamaDecoderLayer):
    def __init__(self, config, layer_idx):
        super(LlamaDecoderLayer, self).__init__()
        self.hidden_size = config.hidden_size

        # self.self_attn = LlamaSdpaAttention(config, layer_idx)
        self.self_attn = LlamaSdpaAttentionDB(config, layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

class LlamaSdpaAttentionDB(LlamaSdpaAttention):
    """
    Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `LlamaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.middleware = {}
        self.transition_matrix = None

    # Adapted from LlamaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )
        # profiler.start()
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]
        
        if self.transition_matrix is None:
            Wk_double = self.k_proj.weight.T.to(torch.double)
            Wv_double = self.v_proj.weight.T.to(torch.double)
            self.transition_matrix = (torch.inverse(Wk_double) @ Wv_double).to(torch.float)
            del Wk_double, Wv_double
        
        attn_output = past_key_value.query(
            query_states, 
            self.layer_idx,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=causal_mask is None and q_len > 1,
            transition_matrix=self.transition_matrix,
            cos=cos,
            sin=sin,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        # profiler.stop()
        # profiler.print()

        # self.middleware.update({
        #     "query_states" : query_states,
        #     "key_states" : key_states,
        #     "value_states" : value_states,
        #     "past_key_value" : past_key_value
        # })

        return attn_output, None, past_key_value
    