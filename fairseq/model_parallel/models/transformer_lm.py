# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from fairseq import utils
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer_lm import (
    base_lm_architecture,
    TransformerLanguageModel,
)
from fairseq.model_parallel.models.transformer import ModelParallelTransformerDecoder
try:
    from fairseq.model_parallel.megatron.mpu import get_model_parallel_group
    from fairseq.model_parallel.megatron.mpu import get_model_parallel_rank
    from fairseq.model_parallel.megatron.mpu import get_model_parallel_world_size
    from fairseq.model_parallel.megatron.mpu import VocabParallelEmbedding
    from fairseq.model_parallel.megatron.mpu.utils import VocabUtility
    has_megatron_submodule = True
except (ImportError, ModuleNotFoundError):
    has_megatron_submodule = False


DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model('model_parallel_transformer_lm')
class ModelParallelTransformerLanguageModel(TransformerLanguageModel):

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        if not has_megatron_submodule:
            raise ImportError(
                '\n\nPlease install the megatron submodule:'
                '\n\n  git submodule update --init '
                'fairseq/model_parallel/megatron'
            )

        # make sure all arguments are present in older models
        base_lm_architecture(args)

        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, 'max_target_positions', None) is None:
            args.max_target_positions = getattr(args, 'tokens_per_sample', DEFAULT_MAX_TARGET_POSITIONS)

        if args.character_embeddings:
            raise NotImplementedError("Character embeddings is not supported for model parallel")
        elif args.adaptive_input:
            raise NotImplementedError("Adaptive input is not supported for model parallel")
        else:
            embed_tokens = cls.build_embedding(args, task.source_dictionary, args.decoder_input_dim)

        decoder = ModelParallelTransformerDecoder(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True,
        )
        return cls(decoder)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        def _vocab_init(tensor, **kwargs):
            nn.init.normal_(tensor, mean=0, std=embed_dim ** -0.5)
            nn.init.constant_(tensor[1], 0)
        embed_tokens = VocabParallelEmbedding(len(dictionary), embed_dim, dictionary.pad(), init_method=_vocab_init)
        return embed_tokens

    def get_normalized_probs(
        self,
        net_output,
        log_probs,
        sample,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = net_output[0]
        vocab_size = len(self.decoder.dictionary)

        if logits.size(-1) == vocab_size:
            # we have the full set of logits
            return super().get_normalized_probs(net_output, log_probs, sample)
        # else: vocab-parallel logits, need to combine them

        assert logits.dim() == 3

        # Get the partition's vocab indices
        get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
        partition_vocab_size = logits.size(-1)
        rank = get_model_parallel_rank()
        world_size = get_model_parallel_world_size()
        vocab_start_index, vocab_end_index = get_vocab_range(
            partition_vocab_size, rank, world_size,
        )

        # Assemble full logits
        full_logits = logits.new_zeros(logits.size(0), logits.size(1), vocab_size)
        full_logits[:, :, vocab_start_index:vocab_end_index] = logits
        torch.distributed.all_reduce(
            full_logits,
            op=torch.distributed.ReduceOp.SUM,
            group=get_model_parallel_group(),
        )

        if log_probs:
            return utils.log_softmax(full_logits, dim=-1)
        else:
            return utils.softmax(full_logits, dim=-1)


@register_model_architecture('model_parallel_transformer_lm', 'transformer_lm_megatron')
def transformer_lm_megatron(args):
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 3072)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 3072 * 4)
    args.decoder_layers = getattr(args, 'decoder_layers', 72)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 32)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_fn = getattr(args, 'activation_fn', 'gelu')
    base_lm_architecture(args)


@register_model_architecture('model_parallel_transformer_lm', 'transformer_lm_megatron_11b')
def transformer_lm_megatron_11b(args):
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 3072)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 3072 * 6)
    args.decoder_layers = getattr(args, 'decoder_layers', 72)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 32)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_fn = getattr(args, 'activation_fn', 'gelu')
    base_lm_architecture(args)
