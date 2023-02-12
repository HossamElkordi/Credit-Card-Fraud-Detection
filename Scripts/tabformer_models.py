import torch
from torch import nn
from transformers import BertTokenizer
from torch.nn import CrossEntropyLoss, LayerNorm
from transformers.modeling_utils import PreTrainedModel
from transformers.models.bert.modeling_bert import ACT2FN
from Scripts.custom_criterion import CustomAdaptiveLogSoftmax
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.models.bert.modeling_bert import BertForMaskedLM
from transformers.models.bert.configuration_bert import BertConfig


class TabFormerBaseModel(PreTrainedModel):
    def __init__(self, hf_model, tab_embeddings, config):
        super().__init__(config)

        self.model = hf_model
        self.tab_embeddings = tab_embeddings

    def forward(self, input_ids, **input_args):
        inputs_embeds = self.tab_embeddings(input_ids)
        return self.model(inputs_embeds=inputs_embeds, **input_args)


class TabFormerHierarchicalLM(PreTrainedModel):
    base_model_prefix = "bert"

    def __init__(self, config, vocab):
        super().__init__(config)

        self.config = config

        self.tab_embeddings = TabFormerEmbeddings(self.config)
        self.tb_model = TabFormerBertForMaskedLM(self.config, vocab)

    def forward(self, input_ids, **input_args):
        inputs_embeds = self.tab_embeddings(input_ids)
        return self.tb_model(inputs_embeds=inputs_embeds, **input_args)


class TabFormerBertLM:
    def __init__(self, special_tokens, vocab, ncols=None, field_hidden_size=768):

        self.ncols = ncols
        self.vocab = vocab
        vocab_file = self.vocab.filename
        hidden_size =  field_hidden_size * self.ncols

        self.config = TabFormerBertConfig(vocab_size=len(self.vocab),
                                          ncols=self.ncols,
                                          hidden_size=hidden_size,
                                          field_hidden_size=field_hidden_size,
                                          flatten=False,
                                          num_attention_heads=self.ncols)

        self.tokenizer = BertTokenizer(vocab_file,
                                       do_lower_case=False,
                                       **special_tokens)
        self.model = self.get_model()

    def get_model(self):
        # hierarchical field CE BERT
        model = TabFormerHierarchicalLM(self.config, self.vocab)
        return model


class TabFormerTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
    ):

        super().__init__(bos_token=bos_token, eos_token=eos_token, unk_token=unk_token)


class TabFormerEmbeddings(nn.Module):
    """TabFormerEmbeddings: Embeds tabular data of categorical variables
        Notes: - All column entries must be integer indices in a vocabolary that is common across columns
        Args:
            config.ncols
            config.num_layers (int): Number of transformer layers
            config.vocab_size
            config.hidden_size
            config.field_hidden_size
        Inputs:
            - **input** (batch, seq_len, ncols): tensor of batch of sequences of rows
        Outputs:
            - **output**: (batch, seq_len, hidden_size): tensor of embedded rows
    """

    def __init__(self, config):
        super().__init__()

        if not hasattr(config, 'num_layers'):
            config.num_layers = 1
        if not hasattr(config, 'nhead'):
            config.nhead = 8

        self.word_embeddings = nn.Embedding(config.vocab_size, config.field_hidden_size,
                                            padding_idx=getattr(config, 'pad_token_id', 0), sparse=False)

        encoder_layer = nn.TransformerEncoderLayer(d_model=config.field_hidden_size, nhead=config.nhead,
                                                   dim_feedforward=config.field_hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        self.lin_proj = nn.Linear(config.field_hidden_size * config.ncols, config.hidden_size)

    def forward(self, input_ids):
        inputs_embeds = self.word_embeddings(input_ids)
        embeds_shape = list(inputs_embeds.size())

        inputs_embeds = inputs_embeds.view([-1] + embeds_shape[-2:])
        inputs_embeds = inputs_embeds.permute(1, 0, 2)
        inputs_embeds = self.transformer_encoder(inputs_embeds)
        inputs_embeds = inputs_embeds.permute(1, 0, 2)
        inputs_embeds = inputs_embeds.contiguous().view(embeds_shape[0:2]+[-1])

        inputs_embeds = self.lin_proj(inputs_embeds)

        return inputs_embeds

class TabFormerBertConfig(BertConfig):
    def __init__(
        self,
        flatten=True,
        ncols=12,
        vocab_size=30522,
        field_hidden_size=64,
        hidden_size=768,
        num_attention_heads=12,
        pad_token_id=0,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.ncols = ncols
        self.field_hidden_size = field_hidden_size
        self.hidden_size = hidden_size
        self.flatten = flatten
        self.vocab_size = vocab_size
        self.num_attention_heads=num_attention_heads

class TabFormerBertForMaskedLM(BertForMaskedLM):
    def __init__(self, config, vocab):
        super().__init__(config)

        self.vocab = vocab
        self.cls = TabFormerBertOnlyMLMHead(config)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            masked_lm_labels=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            lm_labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        sequence_output = outputs[0]  # [bsz * seqlen * hidden]

        if not self.config.flatten:
            output_sz = list(sequence_output.size())
            expected_sz = [output_sz[0], output_sz[1]*self.config.ncols, -1]
            sequence_output = sequence_output.view(expected_sz)
            masked_lm_labels = masked_lm_labels.view(expected_sz[0], -1)

        prediction_scores = self.cls(sequence_output) # [bsz * seqlen * vocab_sz]

        outputs = (prediction_scores,) + outputs[2:]

        # prediction_scores : [bsz x seqlen x vsz]
        # masked_lm_labels  : [bsz x seqlen]

        total_masked_lm_loss = 0

        seq_len = prediction_scores.size(1)
        # TODO : remove_target is True for card
        field_names = self.vocab.get_field_keys(remove_target=True, ignore_special=False)
        for field_idx, field_name in enumerate(field_names):
            col_ids = list(range(field_idx, seq_len, len(field_names)))

            global_ids_field = self.vocab.get_field_ids(field_name)

            prediction_scores_field = prediction_scores[:, col_ids, :][:, :, global_ids_field]  # bsz * 10 * K
            masked_lm_labels_field = masked_lm_labels[:, col_ids]
            masked_lm_labels_field_local = self.vocab.get_from_global_ids(global_ids=masked_lm_labels_field,
                                                                          what_to_get='local_ids')

            nfeas = len(global_ids_field)
            loss_fct = self.get_criterion(field_name, nfeas, prediction_scores.device)

            masked_lm_loss_field = loss_fct(prediction_scores_field.view(-1, len(global_ids_field)),
                                            masked_lm_labels_field_local.view(-1))

            total_masked_lm_loss += 0.0 if torch.isnan(masked_lm_loss_field) else masked_lm_loss_field

        return (total_masked_lm_loss,) + outputs

    def get_criterion(self, fname, vs, device, cutoffs=False, div_value=4.0):

        if fname in self.vocab.adap_sm_cols:
            if not cutoffs:
                cutoffs = [int(vs/15), 3*int(vs/15), 6*int(vs/15)]

            criteria = CustomAdaptiveLogSoftmax(in_features=vs, n_classes=vs, cutoffs=cutoffs, div_value=div_value)

            return criteria.to(device)
        else:
            return CrossEntropyLoss()

class TabFormerBertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = TabFormerBertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class TabFormerBertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = TabFormerBertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class TabFormerBertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.field_hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states