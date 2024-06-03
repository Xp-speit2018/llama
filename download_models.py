import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Model, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torchtext.datasets import WikiText103

def set_proxy():
    os.environ['http_proxy'] = "127.0.0.1:7897"
    os.environ['https_proxy'] = "127.0.0.1:7897"
    # os.environ['http_proxy'] = "127.0.0.1:7898"
    # os.environ['https_proxy'] = "127.0.0.1:7898"

def unset_proxy():
    os.environ.pop('http_proxy')
    os.environ.pop('https_proxy')

def download_GPT2Large(dir):
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-large")
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-large")
    model.save_pretrained(dir)
    tokenizer.save_pretrained(dir)

def download_GPT2XL(dir):
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-xl")
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-xl")
    model.save_pretrained(dir)
    tokenizer.save_pretrained(dir)

def download_BLOOM(dir):
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-7b1")
    model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-7b1")
    model.save_pretrained(dir)
    tokenizer.save_pretrained(dir)

def download_wikitext103():
    load_dataset(path='wikitext', name='wikitext-103-v1').save_to_disk('wikitext-103-v1')

def download_PTB():
    load_dataset("ptb_text_only").save_to_disk("PTB")

def download_wikitext2():
    load_dataset("wikitext", "wikitext-2-raw-v1").save_to_disk("wikitext-2-raw-v1")

def download_llama3B(dir):
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/llama-3B")
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/llama-3B")
    model.save_pretrained(dir)
    tokenizer.save_pretrained(dir)

if __name__ == '__main__':

    set_proxy()
    # download_GPT2XL('./gpt2-xl')
    # download_GPT2Large('./gpt2-large')
    # download_BLOOM('./bloom-7b1')
    download_wikitext103()
    # download_PTB()
    download_wikitext2()
    # download_llama3B('./llama-3B')
    unset_proxy()
