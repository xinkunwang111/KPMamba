import torch

import torch.nn as nn
from linear_attention_transformer import LinearAttentionTransformer
import torch.nn.functional as F


class GatedAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GatedAttention, self).__init__()
        
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.gate = nn.Linear(input_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        batch_size, seq_length, input_dim = x.size()

        
        
        # Linear attention
        linear_output = torch.sigmoid(self.linear(x))  # (batch_size, seq_length, hidden_dim)
        
        # Gated attention
        gate_output = self.sigmoid(self.gate(x))  # (batch_size, seq_length, hidden_dim)
        gated_output = linear_output * gate_output  # Element-wise multiplication
        
        return gated_output

# Example usage
class combined_attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers = 6):
        super(combined_attention, self).__init__()
        #self.gated_attention = GatedAttention(input_dim, hidden_dim)
        #self.linear = nn.Linear(input_dim, hidden_dim)
        #encoder = nn.TransformerEncoderLayer(d_model=96, nhead=8,batch_first=True)

        self.self_attn = nn.ModuleList([nn.TransformerEncoderLayer(d_model=96, nhead=8,batch_first=True) for _ in range(num_layers // 3)])

        #self.gated_attention = nn.ModuleList([GatedAttention(input_dim, hidden_dim) for _ in range(num_layers // 3) ])

        self.lin_attn = LinearAttentionTransformer(dim=96, heads=8,depth = num_layers // 3, max_seq_len = 4096,)

        #self.attn = nn.ModuleList([nn.TransformerEncoderLayer(d_model=96, nhead=8,batch_first=True) for _ in range(num_layers // 2)]+
                                  #[LinearAttentionTransformer(dim=96, heads=8,depth = num_layers // 2, max_seq_len = 4096,)])
                                    
                                    
                                  
                                    
            
    def forward(self, x):

        for layer in self.self_attn:
            x = layer(x) 
        return x

        
        """# Self-attention
        self_attn_output = x
        for layer in self.self_attn:
            self_attn_output = layer(self_attn_output)

        # linear attention  
        linear_output = self.lin_attn(x)

            
        return linear_output + self_attn_output"""
        

        
