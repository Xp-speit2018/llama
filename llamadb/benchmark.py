from modeling_llamadb import LlamaForCausalLMDB
from transformers import AutoTokenizer
from transformers.models.llama import LlamaForCausalLM

import sys 
sys.path.append("..")

from perplexity import perplexity



def ppl():
    model_path = "../llama-2-7b-hf"

    modelDB = LlamaForCausalLMDB.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    device = 'cuda:0'
    root = '../'
    dataset = 'PTB'

    stride = modelDB.config.max_position_embeddings # 4096
    ppl_db = perplexity(modelDB, tokenizer, dataset, device, verbose=True, stride=stride, root=root)
    print(ppl_db)

    del modelDB

    model = LlamaForCausalLM.from_pretrained(model_path)
    stride = model.config.max_position_embeddings # 4096
    ppl_baseline = perplexity(model, tokenizer, dataset, device, verbose=True, stride=stride, root=root)
    print(ppl_baseline)
    del model

if __name__ == "__main__":
    ppl()