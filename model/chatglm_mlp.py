import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM

class chatglm_mlp(nn.Module):
    def __init__(self, model_name, hidden_size=4096):
        super(chatglm_mlp, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(self.device)

        for param in self.model.parameters():
            param.requires_grad = False

        self.layer_norm = nn.LayerNorm(hidden_size).to(self.device)
        output_size = self.model.config.vocab_size
        self.mlp = nn.Linear(output_size, hidden_size).to(self.device)

        self.classifier = nn.Linear(hidden_size, 1).to(self.device)
        self.tanh = nn.Tanh()

    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path, map_location=self.device))
        print(self.down_project.weight.dtype)
        print("Model loaded successfully from", model_path)

    def forward(self, input_ids, attention_mask, generate_text=False):

        input_ids = input_ids.long()
        attention_mask = attention_mask.long()

        transformer_outputs = self.model.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        last_hidden_state = transformer_outputs.hidden_states[-1]
        # print(last_hidden_state.size())
        last_hidden_state = last_hidden_state.to(dtype=torch.float32)

        normalized_output = self.layer_norm(last_hidden_state)
        # last_token_logits = torch.mean(normalized_output, dim=0)
        last_token_logits = torch.mean(normalized_output, dim=0)
        # print(last_token_logits.size())
        tanh_output = self.tanh(last_token_logits)
        classification_output = self.classifier(tanh_output)
        # print(classification_output.size())

        return classification_output, normalized_output