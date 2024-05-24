from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM
import torch
from datasets import load_from_disk, load_dataset
from tqdm import tqdm
import argparse
from time import time

from llama import Tokenizer, Transformer
from torch.nn import CrossEntropyLoss

GPU_MAX_SELECTION_K = 2048

class Timer:
    def __init__(self):
        self.start = time()

    def tick(self):
        return time() - self.start
    
    def reset(self):
        self.start = time()

    def __str__(self):
        return f'[{self.tick():.3f}s] '

t = Timer()

def load_model(model_name, device):
    if model_name == 'gpt2':
        model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    elif model_name == 'gpt2-large':
        model = AutoModelForCausalLM.from_pretrained('gpt2-large').to(device)
        tokenizer = AutoTokenizer.from_pretrained('gpt2-large')
    else:
        raise ValueError('Model not supported')

    return model, tokenizer

def preprocess_dataset(dataset_name, root='./'):
    supported = ['wikitext', 'wikitext-103-v1', 'PTB']
    if dataset_name not in supported:
        raise ValueError('Dataset not supported')
    
    split = 'test'
    # dataset = load_from_disk(root+dataset_name)
    if dataset_name == 'wikitext':
        # dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
        dataset = load_from_disk(root+'wikitext-2-raw-v1')[split]
    elif dataset_name == 'wikitext-103-v1':
        # dataset = load_dataset('wikitext', 'wikitext-103-v1', split=split)
        dataset = load_from_disk(root+'wikitext-103-v1')[split]
    elif dataset_name == 'PTB':
        # dataset = load_dataset('ptb_text_only', 'penn_treebank', split=split).rename_column('sentence', 'text')
        dataset = load_from_disk(root+'PTB')[split].rename_column('sentence', 'text')

    return dataset

def encode_dataset(dataset, tokenizer):
    merged_text = "\n\n".join(dataset['text'])

    if isinstance(tokenizer, Tokenizer):
        # tokenizer is from llama
        token_ids = tokenizer.encode(merged_text, bos=True, eos=True)
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        res = {'input_ids': input_ids, 'attention_mask': attention_mask}
    else:
        # tokenizer is from torch
        res = tokenizer(merged_text, return_tensors='pt')

    return res

def perplexity(model, tokenizer, dataset, device, *, stride=None, verbose=True, debug=False, root='./'):
    model.to(device)
    model.eval()
    if stride is None:
        stride = model.config.n_positions

    if verbose:
        print(f'{t}    Loading dataset {dataset}')
    dataset_pr = preprocess_dataset(dataset, root)
    if verbose:
        print(f'{t}    Encoding dataset')

    encodings = encode_dataset(dataset_pr, tokenizer)

    if isinstance(model, GPT2LMHeadModel):
        max_length = model.config.n_positions
    elif isinstance(model, LlamaForCausalLM):
        max_length = model.config.max_position_embeddings
    else:
        raise ValueError('Model not supported')
    
    # if max_length > GPU_MAX_SELECTION_K:
    #     print(f'{t}Warning: max_length {max_length} is too large for GPU, setting to {GPU_MAX_SELECTION_K}')
    #     max_length = GPU_MAX_SELECTION_K

    # if stride > max_length:
    #     print(f'{t}Warning: stride {stride} is larger than max_length {max_length}, setting to max_length')
    #     stride = max_length

    seq_len = encodings['input_ids'].size(1)
    
    nlls = [] # negative log likelihoods
    total_length = 0
    prev_end_loc = 0
    loss_fn = CrossEntropyLoss()
    for begin_loc in tqdm(range(0, seq_len, stride)):

        # do some padding
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings['input_ids'][:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100 # -100 will be ignored by loss function

        with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                nll = outputs.loss
        
        nlls.append(nll * trg_len)
        total_length += trg_len

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    
    ppl = torch.exp(torch.stack(nlls).sum() / total_length)

    # free memory
    del encodings
    del dataset_pr
    del input_ids
    del target_ids
    del outputs
    torch.cuda.empty_cache()

    return ppl.item()

if __name__ == '__main__':
    t.reset()

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Model name', default='gpt2-large')
    parser.add_argument('--dataset', type=str, help='Dataset name', default='PTB')
    parser.add_argument('--device', type=str, help='Device to use', default='cuda:7')
    parser.add_argument('--verbose', action='store_true', help='Print perplexity after each sample')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--stride', type=int, help='Stride for computing perplexity', default=None)
    args = parser.parse_args()

    # Load model
    print(f'{t}Loading model {args.model}')
    model, tokenizer = load_model(args.model, args.device)

    # Compute perplexity
    print(f'{t}Computing perplexity...')
    if args.stride is None:
        print(f'{t}    Stride not specified, using default stride {model.config.n_positions}')
        args.stride = model.config.n_positions
    if args.verbose:
        print(f'{t}    Using device {args.device}, stride {args.stride} over {model.config.n_positions} tokens per input')
        print(f'{t}    For stride meaning check https://huggingface.co/docs/transformers/perplexity')
    ppl = perplexity(model, tokenizer, args.dataset, args.device, stride=args.stride, verbose=args.verbose, debug=args.debug)

    # Print results
    print(f'{t}Perplexity: {ppl}')
