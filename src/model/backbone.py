
import torch.nn as nn
from transformers import BertModel, BertConfig
import torch

class Bertax(nn.Module):
    def __init__(self,num_layers=8,
            num_attention_heads=4,
            hidden_size=512,
            intermediate_size=2048,
            vocab_size=69,
            max_position_embeddings=22,
            num_classes=10,
            dropout_rate=0.1):

        super(Bertax, self).__init__()

        # Initialized to pretrain mode
        self.mode = "pretrain"

        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            hidden_dropout_prob=dropout_rate,
            attention_probs_dropout_prob=dropout_rate
        )

        self.bert = BertModel(config)

        self.mlm_head = nn.Linear(hidden_size, vocab_size)

        #TODO: I want to make a wrapper class for the nn.sequential object, so I can factor it out from this model. Help me write that calss. 
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2 ),
            nn.ReLU(),
            nn.Dropout(p = dropout_rate),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def preTrainMode(self):
        self.mode = "pretrain"

    def classifyMode(self):
        self.mode = "classify"

    def forward(self, input_ids, attention_mask = None):
        outputs = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask
            )
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output

        if self.mode == "pretrain":
            #USE MLM-head for pre-training
            return self.mlm_head(sequence_output)
        if self.mode == "classify":
                return self.classification_head(pooled_output)
        else:
             raise ValueError(f"Invalid mode: {self.mode}. Use 'pretrain' or 'classify'.")
        
class ModularBertax(nn.Module):
    def __init__(self,encoder, mlm_head, classification_head):

        super(ModularBertax, self).__init__()

        # Initialized to pretrain mode
        self.mode = "pretrain"
        
        self.bert = encoder
        self.mlm_head = mlm_head
        self.classification_head = classification_head

    def preTrainMode(self):
        self.mode = "pretrain"

    def classifyMode(self):
        self.mode = "classify"

    def forward(self, input_ids, attention_mask = None):
        outputs = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask
            )
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output

        if self.mode == "pretrain":
            #USE MLM-head for pre-training
            return self.mlm_head(sequence_output)
        if self.mode == "classify":
                return self.classification_head(pooled_output)
        else:
             raise ValueError(f"Invalid mode: {self.mode}. Use 'pretrain' or 'classify'.")
        
class OverlappingKmerModularBertax(nn.Module):
    def __init__(self, encoder, mlm_head, classification_head):
        super(OverlappingKmerModularBertax, self).__init__()
        
        # Keep the same structure/fields as your original ModularBertax
        self.mode = "pretrain"
        
        self.bert = encoder
        self.mlm_head = mlm_head
        self.classification_head = classification_head

    def preTrainMode(self):
        self.mode = "pretrain"

    def classifyMode(self):
        self.mode = "classify"

    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids (torch.Tensor): Shape [k, seq_len], i.e. multiple k-mers for a single example.
            attention_mask (torch.Tensor): Same shape [k, seq_len].
        
        Returns:
            In preTrainMode: MLM logits from self.mlm_head.
            In classifyMode: Hierarchical classification logits from self.classification_head.
        """

        if self.mode == "pretrain":
            # 1) Randomly pick ONE of the k input sequences
            #    (You could also pick an index using `random.randint(...)` or `torch.randint`.)
            rand_index = torch.randint(0, input_ids.shape[0], (1,)).item()

            chosen_input_ids = input_ids[rand_index].unsqueeze(0)      # [1, seq_len]
            chosen_attention_mask = None
            if attention_mask is not None:
                chosen_attention_mask = attention_mask[rand_index].unsqueeze(0)  # [1, seq_len]

            # 2) Pass the chosen sequence to BERT
            outputs = self.bert(
                input_ids=chosen_input_ids,
                attention_mask=chosen_attention_mask
            )
            sequence_output = outputs.last_hidden_state  # shape [1, seq_len, hidden_size]

            # 3) Return MLM logits (like the old `ModularBertax`)
            return self.mlm_head(sequence_output)

        elif self.mode == "classify":
            # 1) Loop over each of the k input sequences
            pooled_outputs = []
            for i in range(input_ids.shape[0]):
                # shape: [1, seq_len]
                this_input = input_ids[i].unsqueeze(0)
                this_attention = attention_mask[i].unsqueeze(0) if attention_mask is not None else None

                outputs = self.bert(
                    input_ids=this_input,
                    attention_mask=this_attention
                )
                #print(f"Output {i}: {outputs}")
                # Collect the pooler_output (CLS embedding)
                pooled_outputs.append(outputs.pooler_output)  # each [1, hidden_size]
            #print(f"pooled_outputs: {pooled_outputs}")
            # 2) Aggregate the CLS outputs
            #    For example, take the mean over k
            #    Result: shape [1, hidden_size]
            stacked_pools = torch.cat(pooled_outputs, dim=0)   # shape: [k, hidden_size]
            pooled_output = stacked_pools.mean(dim=0, keepdim=True)  # => [1, hidden_size]

            # 3) Pass aggregated CLS embedding to classification head
            return self.classification_head(pooled_output)

        else:
            raise ValueError(f"Invalid mode: {self.mode}. Use 'pretrain' or 'classify'.")
                




         
         
