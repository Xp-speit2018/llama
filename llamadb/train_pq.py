from faiss import IndexPQ, METRIC_INNER_PRODUCT, index_factory, write_index
import numpy as np
from typing import List, Tuple
import json
import itertools
import tqdm

class RecursiveNamespace:
    def __init__(self):
        self.__dict__['_attributes'] = {}

    def __getattr__(self, name):
        if name not in self._attributes:
            self._attributes[name] = RecursiveNamespace()
        return self._attributes[name]

    def __setattr__(self, name, value):
        if isinstance(value, RecursiveNamespace):
            self._attributes[name] = value
        else:
            current = self
            *parts, last = name.split('.')
            for part in parts:
                current = getattr(current, part)
            current._attributes[last] = value

    def __repr__(self):
        return repr(self._attributes)
    
    @staticmethod
    def from_dict(d):
        ns = RecursiveNamespace()
        for k, v in d.items():
            if isinstance(v, dict):
                ns._attributes[k] = RecursiveNamespace.from_dict(v)
            else:
                ns._attributes[k] = v
        return ns
    
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
            
def load_config(filePath):
    '''
    load configuration json to a RecursiveNamespace object
    '''
    with open(filePath, 'r') as f:
        return RecursiveNamespace.from_dict(json.loads(f.read()))
    
    
if __name__ == "__main__":
    config = load_config('../llama-2-7b-hf/config.json')
    
    M = 16
    nbits = 10
    dim = int(config.hidden_size / config.num_attention_heads)

    iter_comb = list(itertools.product(range(config.num_hidden_layers), range(config.num_attention_heads)))

    for i, j in tqdm.tqdm(iter_comb):

        # index = index_factory(dim, f'PQ{M}x{nbits}', METRIC_INNER_PRODUCT)
        # index.train(
        #     read_fvecs(f'../llama_key/PTB/key_{i}_{j}.fvecs')
        # )
        # ivf = index_factory(dim, f"IVF1,PQ{M}x{nbits}", METRIC_INNER_PRODUCT)
        # ivf.quantizer.add(np.zeros((1, dim)))
        # ivf.pq = index.pq
        # ivf.is_trained = True
        # write_index(ivf, f'../llama_pqindex/PTB/key_{i}_{j}.ivf')
        index = IndexPQ(dim, M, nbits, METRIC_INNER_PRODUCT)
        index.train(
            read_fvecs(f'../llama_key/PTB/key_{i}_{j}.fvecs')
        )
        write_index(index, f'../llama_pqindex/PTB/key_{i}_{j}.pq')