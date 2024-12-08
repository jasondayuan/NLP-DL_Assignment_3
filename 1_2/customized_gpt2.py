from turtle import forward
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block, GPT2Model, GPT2LMHeadModel

class CustomizedGPT2Attention(GPT2Attention):
    """
    GPT2 flash attention module. This module inherits from `GPT2Attention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cache_key = None
        self.cache_value = None

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        attention_mask: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        assert 'cached' in kwargs, "[NLPDL ERROR] CustomizedGPT2Attention: `cached` argument does not exist"
        cached = kwargs['cached']

        if not cached:
            self.cache_key = None
            self.cache_value = None

            # Prepare query, key, value matrix
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2) # each of them has shape (batch_size, seq_len, dim)
            self.cache_key = key # (B, T, C)
            self.cache_value = value

            query = self._split_heads(query, self.num_heads, self.head_dim) # [batch_size, num_heads, seq_len, head_dim]
            key = self._split_heads(key, self.num_heads, self.head_dim) # [batch_size, num_heads, seq_len, head_dim]
            value = self._split_heads(value, self.num_heads, self.head_dim) # [batch_size, num_heads, seq_len, head_dim]

            # Self-attention mechanism
            attn_output, attn_weights = self._attn(query, key, value, attention_mask)

        else:
            query, key, value = self.c_attn(hidden_states[:, -1, :].unsqueeze(1)).split(self.split_size, dim=2) # (B, 1, C)
            self.cache_key = torch.cat([self.cache_key, key], dim=1) # (B, T, C)
            self.cache_value = torch.cat([self.cache_value, value], dim=1)

            query = self._split_heads(query, self.num_heads, self.head_dim)
            key = self._split_heads(self.cache_key, self.num_heads, self.head_dim)
            value = self._split_heads(self.cache_value, self.num_heads, self.head_dim)

            attn_output, attn_weights = self._attn(query, key, value, attention_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output


class CustomizedGPT2Block(GPT2Block):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx=layer_idx)
        self.attn = CustomizedGPT2Attention(config=config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        attention_mask: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        assert 'cached' in kwargs, "[NLPDL ERROR] CustomizedGPT2Block: `cached` argument does not exist"
        cached = kwargs['cached']

        residual = hidden_states

        # self-attention (class `CustomizedGPT2AttentionWithFasterCache`)
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            cached = cached
        )

        # residual connection
        hidden_states = attn_output + residual


        residual = hidden_states

        # feed-forward
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states


        return hidden_states


class CustomizedGPT2Model(GPT2Model):
    def __init__(self, config):
        super().__init__(config)
        self.h = nn.ModuleList([CustomizedGPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self._attn_implementation = config._attn_implementation
        assert self._attn_implementation == 'eager', "[NLPDL ERROR] set _attn_implementation to either 'eager' or 'faster_cache' in this version"

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:

        assert 'cached' in kwargs, "[NLPDL ERROR] CustomizedGPT2Model: `cached` argument does not exist"
        cached = kwargs['cached']

        input_shape = input_ids.size()
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Prepare input embeddings
        inputs_embeds = self.wte(input_ids)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        # Prepare Attention mask
        attention_mask = attention_mask.view(batch_size, -1) if attention_mask is not None else None
        attention_mask = attention_mask[:, None, None, :]
        attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min # the padding tokens are masked(set to -inf)

        hidden_states = self.drop(hidden_states)
        if not cached:
            output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),) # (-1, seq_len, hidden_size)
        else:
            output_shape = (-1,) + (1,) + (hidden_states.size(-1),) # (-1, 1, hidden_size)

        # Iterate over all GPT2 layer, i.e. `block`
        for i, block in enumerate(self.h):
            outputs = block(
                hidden_states if not cached else hidden_states[:, -1, :].unsqueeze(1),
                attention_mask=attention_mask,
                cached=cached
            )

            hidden_states = outputs


        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)

        return hidden_states


class CustomizedGPT2LMHeadModel(GPT2LMHeadModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = CustomizedGPT2Model(config)

        # KV cache utils
        self.prev_input_ids = None
        self.prev_attention_mask = None

        # Initialize weights and apply final processing
        self.post_init()

    def clear_cache(self):
        for block in self.transformer.h:
            block.attn.cache_key = None
            block.attn.cache_value = None
        self.prev_input_ids = None
        self.prev_attention_mask = None

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
    ):
        
        # Check if cached
        cached = (self.prev_input_ids != None) and torch.equal(input_ids[:, :-1], self.prev_input_ids) and torch.equal(attention_mask[:, :-1], self.prev_attention_mask)
        if not cached:
            self.clear_cache()
            self.prev_input_ids = input_ids
            self.prev_attention_mask = attention_mask
        
        hidden_states = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            cached=cached
        )

        lm_logits = self.lm_head(hidden_states)

        return {
            'logits': lm_logits,
        }