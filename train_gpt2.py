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
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        
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
        self.attn = CausalSelfAttention(config)
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
    

class GPT(nn.Module):
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
    
    def forward(self, idx):
        #idx to be shaped [B, T] where B is the batch size and T is the sequence length
        B, T = idx.size()
        assert T <= self.config.block_size, "Cannot forward sequence length of length {} > block size {}".format(T, self.config.block_size)
        
        #forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) #shape [T]
        pos_emb = self.transformer.wpe(pos) #shape [T, n_embd]
        tok_emb = self.transformer.wte(idx) #shape [B, T, n_embd]
        x = tok_emb + pos_emb
        
        #transformer blocks forwarded
        for block in self.transformer.h:
            x = block(x)
        #final layer normalization
        x = self.transformer.ln_f(x)
        
        #get logits
        logits = self.lm_head(x) #shape [B, T, vocab_size]
        return logits #probability distribution over the vocabulary 
        
        
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
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        
        #copy the weights from the pretrained model to the from scratch model
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # ignore the buffer keys
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these too
        
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        
        assert len(sd_keys_hf) == len(sd_keys), f"mismatch in number of keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                #special the weights that need to be transposed
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].T)
            else:
                #vanilla copy over the rest of the weights
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
                    
        return model
    
# ------------------------INFERENCE-------------------------------- #
#autodetect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_return_sequences = 5
max_length = 30

## model = GPT.from_pretrained('gpt2') #pretrained
model = GPT(GPTConfig()) #Initialize at random
model.eval()
model.to(device)


#prefix tokens
import tiktoken #from openai

#get the encoding for the gpt2 tokenizer
enc = tiktoken.get_encoding('gpt2')

#encode the input tokens
tokens = enc.encode("Hello, I'm a language model,")

#convert the tokens to a torch.long tensor
tokens = torch.tensor(tokens, dtype=torch.long) # (n_tokens,) #in this case 8 tokens

#repeat the tokens for the number of sequences to generate
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (n_sequences, n_tokens) #in this case 5 sequences of 8 tokens

#move the tokens to the GPU
x = tokens.to(device)

#Now we're all set up to generate some text

#set the seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)

#generate the text
while x.size(1) < max_length:
    #forward the model to get the logits
    with torch.no_grad(): #no need to compute gradients
        logits = model(x) # (B, T, vocab_size) == (n_sequences=5, n_tokens=8, vocab_size=50257)
        #get the logits for the last token for next token prediction
        logits = logits[:, -1, :] # (B, vocab_size) == (n_sequences=5, vocab_size=50257)
        #get the probabilities
        probs = F.softmax(logits, dim=-1)
        #do top k sampling to get the next token (in this case of 50 as hf default)
        #top k probs here would be (5, 50), top k indices would be (5, 50)
        top_k_probs, top_k_indices = torch.topk(probs, 50, dim=-1)
        #sample from the top k probs
        ix = torch.multinomial(top_k_probs, num_samples=1) # (B, 1) == (n_sequences=5, 1)
        #get the token ids
        xcol = torch.gather(top_k_indices, -1, ix) # (B, 1) == (n_sequences=5, 1)
        #concatenate the new token to the input
        x = torch.cat((x, xcol), dim=1)


#print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)     
        
        