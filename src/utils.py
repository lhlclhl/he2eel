import torch
from pytorch_lightning.metrics import Metric


class MicroF1(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("n", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("prec_d", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("rec_d", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, p, g):

        self.n += len(g.intersection(p))
        self.prec_d += len(p)
        self.rec_d += len(g)

    def compute(self):
        p = self.n.float() / self.prec_d
        r = self.n.float() / self.rec_d
        return (2 * p * r / (p + r)) if (p + r) > 0 else (p + r)


class MacroF1(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("n", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("d", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, p, g):

        prec = len(g.intersection(p)) / len(p) if p else 0
        rec = len(g.intersection(p)) / len(g) if g else 0

        self.n += (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else (prec + rec)
        self.d += 1

    def compute(self):
        return (self.n / self.d) if self.d > 0 else self.d


class MicroPrecision(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("n", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("d", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, p, g):
        self.n += len(g.intersection(p))
        self.d += len(p)

    def compute(self):
        return (self.n.float() / self.d) if self.d > 0 else self.d


class MacroPrecision(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("n", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("d", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, p, g):
        self.n += len(g.intersection(p)) / len(p) if p else 0
        self.d += 1

    def compute(self):
        return (self.n / self.d) if self.d > 0 else self.d


class MicroRecall(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("n", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("d", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, p, g):
        self.n += len(g.intersection(p))
        self.d += len(g)

    def compute(self):
        return (self.n.float() / self.d) if self.d > 0 else self.d


class MacroRecall(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("n", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("d", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, p, g):
        self.n += len(g.intersection(p)) / len(g) if g else 0
        self.d += 1

    def compute(self):
        return (self.n / self.d) if self.d > 0 else self.d


def get_markdown(sentences, entity_spans):
    return_outputs = []
    for sent, entities in zip(sentences, entity_spans):
        text = ""
        last_end = 0
        for begin, end, href in entities:
            text += sent[last_end:begin]
            text += "[{}](https://en.wikipedia.org/wiki/{})".format(
                sent[begin:end], href.replace(" ", "_")
            )
            last_end = end

        text += sent[last_end:]
        return_outputs.append(text)

    return return_outputs


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


from torch import nn
from transformers import LongformerForMaskedLM
from transformers.models.longformer.modeling_longformer import LongformerModel, LongformerLMHead, LongformerEmbeddings, LongformerEncoder, LongformerPooler

def create_position_ids_from_input_ids(input_ids, padding_idx):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
    return incremental_indices.long() + padding_idx
class LongformerForMaskedLM1(LongformerForMaskedLM):
    def __init__(self, config):
        super(LongformerForMaskedLM, self).__init__(config)

        self.longformer = LongformerModel1(config, add_pooling_layer=False)
        self.lm_head = LongformerLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()
class LongformerModel1(LongformerModel):
    def __init__(self, config, add_pooling_layer=True):
        super(LongformerModel, self).__init__(config)
        self.config = config

        if isinstance(config.attention_window, int):
            assert config.attention_window % 2 == 0, "`config.attention_window` has to be an even value"
            assert config.attention_window > 0, "`config.attention_window` has to be positive"
            config.attention_window = [config.attention_window] * config.num_hidden_layers  # one value per layer
        else:
            assert len(config.attention_window) == config.num_hidden_layers, (
                "`len(config.attention_window)` should equal `config.num_hidden_layers`. "
                f"Expected {config.num_hidden_layers}, given {len(config.attention_window)}"
            )

        self.embeddings = LongformerEmbeddings1(config)
        self.encoder = LongformerEncoder(config)
        self.pooler = LongformerPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()
    def add_new_word_embeddings(self, vocab_size):
        self.embeddings.new_word_embeddings = nn.Embedding(
            vocab_size, 
            self.embeddings.word_embeddings.embedding_dim
        )
class LongformerEmbeddings1(LongformerEmbeddings):
    def __init__(self, config):
        super(LongformerEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.new_word_embeddings = None
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )
    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx).to(input_ids.device)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            if self.new_word_embeddings is None:
                inputs_embeds = self.word_embeddings(input_ids)
            else:
                emb_size = self.word_embeddings.num_embeddings
                nw_mask = (input_ids >= emb_size).int()
                inputs_embeds = self.word_embeddings(input_ids*(1-nw_mask))
                inputs_embeds_nw = self.new_word_embeddings((input_ids-emb_size)*nw_mask)
                inputs_embeds = inputs_embeds * (1-nw_mask.unsqueeze(-1)) + \
                                inputs_embeds_nw * nw_mask.unsqueeze(-1)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
