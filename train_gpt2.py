from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        """Causal self-attention mechanism"""
        
        #Make sure the number of hidden units is divisible by the number of attention heads
        assert config.n_embd % config.n_head == 0 
        
        #key, query, value projections for all heads in a batch
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3) #784 x 2352
        
        #output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        #regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        #a mask following HF/OAI naming, torch.tril returns the lower triangular part of a matrix with the main diagonal
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)
                                    .view(1, 1, config.block_size, config.block_size)))
        
    def forward(self, x):
        B, T, C = x.size() #batch size, sequence length, number of hidden units (n_embd)
        #key, query, value projections for all heads in a batch
        qkv = self.c_attn(x)
        
        #separate the key, query, value projections for all heads in a batch
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) #[B X n_head x T x hs] (b=batch size, n_head=number of heads, T=sequence length, hs=hidden size)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) #[B X n_head x T x hs]
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) #[B X n_head x T x hs]
        
        #atention (processes the computations of the large T,T matrix for all the q's and k's)
        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5)) # qk/sqrt(d_k)
        
        #masking the upper triangular part of the matrix for the causality 
        # (Causalty here means that the model can only attend to the past tokens,
        # thats why the mask is applied to the upper triangular part of the matrix,
        # which conforms the already seen tokens)
        att = att.masked_fill(self.bias[:, :, T, T], float('-inf'))
        
        #softmax
        att = F.softmax(att, dim=-1)
        
        #Now attend the values
        y = att @ v # [B X n_head x T,T] @ [B X n_head x T x hs] = [B X n_head x T x hs]
        
        #reassemble all head outputs side by side to get the expected shape    
        y = y.transpose(1, 2).contiguous().view(B, T, C) # [B X T X n_head x hs] -> [B X T X hs]
        
        #output projection, this adds a final bit of complexity to the model
        y = self.c_proj(y)
        
        return y
      

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        """MLP for the transformer block"""
        
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4) #784 x 3072
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd) #3072 x 784
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x) 
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        """Transformer block"""
        
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = nn.CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        
    def forward(self, x):
        x += self.attn(self.ln_1(x))
        x += self.mlp(self.ln_2(x))
        return x
    
    
@dataclass
class GPTConfig:
    block_size: int = 1024 #max sequence length
    vocab_size: int = 50257 #number of tokens in the vocabulary: 50000 BPE merges + 256 bytes tokens + 1 <endoftext> token
    n_layer: int = 12 #number of transformer blocks (or layers)
    n_head: int = 6 #number of attention heads per block
    n_embd: int = 768 #number of hidden units in the transformer blocks
    

class GPT(nn.module):
    def __init__(self, config):
        super().__init__()
        """Skeleton of the GPT2 model"""
        
        # Store the config
        self.config = config
        
        #Transformer model
        self.transformer = nn.ModuleDict(dict(
            
            #weights of the token embeddings
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            
            #weights of the position embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd),
            
            #hidden layers of the transformer (Block is a class that defines a transformer block with n_head attention heads
            #and n_embd hidden units along with layer normalization and residual connections (if applicable))
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            
            #final layer normalization
            ln_f = nn.LayerNorm(config.n_embd),
            
        ))
        
        #classification head to predict
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained gpt2 from huggingface"""
        assert model_type in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
        from transformers import GPT2LMHeadModel
        print(F"Loading pretrained model {model_type} from Huggingface model hub")
        
        #number of layers, hidden units and attention heads for each model are predetermined by model_type
        config_args ={
            'gpt2': dict(n_layer=12, n_embd=768, n_head=12), #124M parameters
            'gpt2-medium': dict(n_layer=24, n_embd=1024, n_head=16), #350M parameters
            'gpt2-large': dict(n_layer=36, n_embd=1280, n_head=20), #774M parameters
            'gpt2-xl': dict(n_layer=48, n_embd=1600, n_head=25), #1558M parameters
        }[model_type]
        
        config_args['vocab_size'] = 50257 #all gpt2 models have the same vocabulary size
        config_args['block_size'] = 1024 #all gpt2 models have the same sequence length
        
        #create a from scratch initialized gpt2 model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        
        #discard the buffer keys that are not present in the pretrained model
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  
        
        #initialize huggingface model
        model_hf = 