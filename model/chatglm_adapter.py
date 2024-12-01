import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from peft import inject_adapter_in_model, LoraConfig
import json

class chatglm_adapter(nn.Module):
    def __init__(self, model_name, adapter_size=64, hidden_size=4096):
        super(chatglm_adapter, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=4,
            lora_alpha=8,
            lora_dropout=0.5
        )
        self.model_lora = get_peft_model(self.model, peft_config).to(self.device)

        for param in self.model.parameters():
            param.requires_grad = False
        # print(self.model.config)
        input_size = self.model.config.hidden_size
        self.down_project = nn.Linear(input_size, adapter_size).to(self.device)
        self.non_linear_func = nn.ReLU()
        self.up_project = nn.Linear(adapter_size, input_size).to(self.device)

        self.layer_norm = nn.LayerNorm(hidden_size).to(self.device)
        output_size = self.model.config.vocab_size
        self.mlp = nn.Linear(output_size, hidden_size).to(self.device)
        # self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        self.classifier = nn.Linear(hidden_size, 1).to(self.device)
        self.tanh = nn.Tanh()

    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path, map_location=self.device))
        print(self.down_project.weight.dtype)
        print("Model loaded successfully from", model_path)

    def forward(self, input_ids, attention_mask, generate_text=False):
        # print(input_ids[0][0:50])
        input_ids = input_ids.long()
        # print(input_ids[0][0:50])
        attention_mask = attention_mask.long()
        # print(attention_mask[0][0:50])

        transformer_outputs = self.model.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        last_hidden_state = transformer_outputs.hidden_states[-1]
        # print(last_hidden_state[0][0][0:50])
        last_hidden_state = last_hidden_state.to(dtype=torch.float32)

        # adapter
        down_projected = self.down_project(last_hidden_state)
        # print(down_projected[0][0][0:50])
        non_linear = self.non_linear_func(down_projected)
        # print(non_linear[0][0][0:50])
        up_projected = self.up_project(non_linear)
        # print(up_projected[0][0][0:50])
        adapted_output = last_hidden_state + up_projected
        # print(adapted_output[0][0][0:50])
        # print(adapted_output.size())
        # torch.Size([3000, 4, 4096])
        # print(input_ids)


        normalized_output = self.layer_norm(adapted_output)

        last_token_logits = torch.mean(normalized_output, dim=0)
        # print(last_token_logits[0][0:50])
        # torch.Size([4, 4096])
        # print(last_token_logits.size())
        # print(last_token_logits)
        tanh_output = self.tanh(last_token_logits)
        # print(tanh_output)
        classification_output = self.classifier(tanh_output)

        return classification_output, normalized_output
