"""
Interventions on attention heads: mean ablation, and path ablation through Q, K or V.

**Mean ablation** replaces a head's output (`attn.z`) with its average over held-out
inputs. The head still adds its typical offset to the residual stream, but carries no
information about this particular input.

**Path ablation** asks how two heads compose. In a two-layer attention-only model, layer
1 reads the residual stream three times, to form its queries, keys and values. The stream
it reads is the embeddings plus what each layer-0 head wrote. Path ablation mean-ablates a
layer-0 head in just one of those three reads, leaving the other two, and the direct path
to the logits, untouched. An induction head should need the previous-token head in its
keys (K-composition: "the token before me was x") and not in its queries or values.

Layer 1's attention is recomputed here from the edited stream, mirroring tiny-transformer's
Attention.forward, and the result is patched in. With nothing ablated, the recomputation
must reproduce the clean pattern exactly; a test checks it.

**Composition scores** (Elhage et al. 2021) measure the same thing from the weights alone:
how much of what a layer-0 head writes (its OV circuit) a layer-1 head's QK or OV circuit
can read.
"""

import math
from typing import Literal

import einops
import torch
from jaxtyping import Float, Int
from tiny_transformer.model import Attention, Block, Patch, Transformer
from torch import Tensor

QKV = Literal['q', 'k', 'v']


def head_means(cache: dict[str, Tensor], n_layers: int) -> dict[int, Float[Tensor, 'head d_head']]:
  """Each head's average output over every sequence and position of a cached batch."""
  return {layer: cache[f'blocks.{layer}.attn.z'].mean(dim=(0, 1)) for layer in range(n_layers)}


def mean_ablation(heads: list[tuple[int, int]], means: dict[int, Tensor]) -> dict[str, Patch]:
  """Patches that replace each listed (layer, head)'s output with its mean."""
  by_layer: dict[int, list[int]] = {}
  for layer, head in heads:
    by_layer.setdefault(layer, []).append(head)

  def ablate(layer: int, chosen: list[int]) -> Patch:
    def patch(z: Tensor) -> Tensor:
      z = z.clone()
      z[:, :, chosen] = means[layer][chosen].to(z.dtype)
      return z
    return patch

  return {f'blocks.{layer}.attn.z': ablate(layer, chosen) for layer, chosen in by_layer.items()}


def identity(n_layers: int) -> dict[str, Patch]:
  """The control: patches every head's output with itself, which must change nothing."""
  return {f'blocks.{layer}.attn.z': (lambda z: z) for layer in range(n_layers)}


def _blocks(model: Transformer) -> tuple[Block, Block]:
  if model.cfg.n_layers != 2:
    raise ValueError(f'path ablation is defined for two-layer models, not {model.cfg.n_layers}')
  first, second = model.blocks
  assert isinstance(first, Block) and isinstance(second, Block)
  return first, second


def head_writes(attn: Attention, z: Float[Tensor, 'batch seq head d_head']) -> Float[Tensor, 'batch seq head d_model']:
  """What each head adds to the residual stream: its slice of W_O applied to its output."""
  W_o = einops.rearrange(attn.W_o.weight, 'm (h d) -> h d m', h=attn.n_heads)
  return einops.einsum(z, W_o, 'b s h d, h d m -> b s h m')


def path_ablation(
  model: Transformer, tokens: Int[Tensor, 'batch seq'], senders: list[int], means: Float[Tensor, 'head d_head'], path: QKV
) -> dict[str, Patch]:
  """
  Patches that mean-ablate layer-0 `senders` only where layer 1 reads its `path` input.

  Runs a clean forward pass to get the residual stream entering layer 1, swaps the
  senders' writes for their mean writes, and recomputes layer 1's attention with the
  edited stream feeding queries, keys or values alone.
  """
  first, second = _blocks(model)
  _, cache = model(tokens, return_cache=True)
  resid = cache['blocks.1.resid_pre']
  z = cache['blocks.0.attn.z']
  edited_z = z.clone()
  edited_z[:, :, senders] = means[senders].to(z.dtype)
  # The stream minus the senders' real writes plus their mean writes. W_O's bias is shared
  # by every input, so it cancels.
  delta = (head_writes(first.attn, edited_z) - head_writes(first.attn, z))[:, :, senders].sum(dim=2)
  clean_in = second.ln1(resid)
  edited_in = second.ln1(resid + delta)
  inputs = {name: edited_in if name == path else clean_in for name in ('q', 'k', 'v')}
  pattern, new_z = attention(second.attn, inputs['q'], inputs['k'], inputs['v'])
  if path == 'v':
    return {'blocks.1.attn.z': new_z}
  return {'blocks.1.attn.pattern': pattern}


def attention(
  attn: Attention, q_in: Tensor, k_in: Tensor, v_in: Tensor
) -> tuple[Float[Tensor, 'batch head query key'], Float[Tensor, 'batch seq head d_head']]:
  """Attention.forward's pattern and head outputs, with separate inputs for Q, K and V."""
  seq = q_in.shape[1]
  q = einops.rearrange(attn.W_q(q_in), 'b s (h d) -> b h s d', h=attn.n_heads)
  k = einops.rearrange(attn.W_k(k_in), 'b s (h d) -> b h s d', h=attn.n_heads)
  v = einops.rearrange(attn.W_v(v_in), 'b s (h d) -> b h s d', h=attn.n_heads)
  scores = einops.einsum(q, k, 'b h q d, b h k d -> b h q k') / math.sqrt(attn.d_head)
  scores = scores.masked_fill(~attn.causal_mask[:seq, :seq], float('-inf'))
  pattern = scores.softmax(dim=-1)
  return pattern, einops.einsum(pattern, v, 'b h q k, b h k d -> b q h d')


def composition_scores(model: Transformer) -> dict[str, Float[Tensor, 'sender receiver']]:
  """
  Q-, K- and V-composition between every layer-0 head (rows) and layer-1 head (columns).

  Score = ‖A·B‖_F / (‖A‖_F ‖B‖_F), where B is the layer-0 head's OV circuit (what it
  writes, given what it reads) and A is the matrix of the layer-1 circuit that consumes it:
  W_Qᵀ W_K for K-composition, W_Kᵀ W_Q for Q-composition, W_O W_V for V-composition.
  Layer 1's LayerNorm sits in between. Its gain is folded into W_Q, W_K and W_V and its
  centring kept as a projection; its division by the norm only rescales, which the ratio
  ignores. Compare against an untrained model: random matrices score well above zero.
  """
  first, second = _blocks(model)
  d_model, d_head, n_heads = model.cfg.d_model, model.cfg.d_head, model.cfg.n_heads
  with torch.no_grad():
    norm = torch.eye(d_model, device=first.attn.W_o.weight.device)
    if isinstance(second.ln1, torch.nn.LayerNorm | torch.nn.RMSNorm):
      gain = second.ln1.weight if second.ln1.weight is not None else torch.ones(d_model)
      if isinstance(second.ln1, torch.nn.LayerNorm):
        norm = norm - 1 / d_model  # centring: x - mean(x)
      norm = torch.diag(gain) @ norm

    def per_head(weight: Tensor) -> Tensor:  # [d_model out, d_model in] → [head, d_head, d_model]
      return einops.rearrange(weight, '(h d) m -> h d m', h=n_heads)

    ov0 = torch.stack([
      first.attn.W_o.weight[:, h * d_head:(h + 1) * d_head] @ per_head(first.attn.W_v.weight)[h] for h in range(n_heads)
    ])  # [head, d_model, d_model]
    q1 = per_head(second.attn.W_q.weight) @ norm
    k1 = per_head(second.attn.W_k.weight) @ norm
    v1 = per_head(second.attn.W_v.weight) @ norm
    o1 = einops.rearrange(second.attn.W_o.weight, 'm (h d) -> h m d', h=n_heads)
    readers = {
      'k': q1.transpose(1, 2) @ k1,  # W_Qᵀ W_K: the key side reads the stream
      'q': k1.transpose(1, 2) @ q1,  # W_Kᵀ W_Q: the query side reads it
      'v': o1 @ v1,
    }
    scores = {}
    for name, reader in readers.items():
      product = einops.einsum(reader, ov0, 'r i j, s j k -> s r i k')
      scores[name] = (
        torch.linalg.matrix_norm(product)
        / torch.linalg.matrix_norm(reader)[None, :]
        / torch.linalg.matrix_norm(ov0)[:, None]
      ).cpu()
  return scores
