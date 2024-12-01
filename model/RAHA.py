import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig

class ChatGLM_TEA(nn.Module):
    def __init__(self, model_name, adapter_size=64, hidden_size=65024):
        super(ChatGLM_TEA, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.hidden = hidden_size
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(self.device)

        # Freeze all parameters in the base model to prevent them from being updated during training
        for param in self.model.parameters():
            param.requires_grad = False

        config = LoraConfig(
            target_modules=[
                # glm
                "transformer.encoder.layers.27.mlp.dense_h_to_4h",
                "transformer.encoder.layers.27.mlp.dense_4h_to_h",

                # bloom
                # "transformer.h.23.mlp.dense_h_to_4h",
                # "transformer.h.23.mlp.dense_4h_to_h",

                # llama
                # "model.layers.31.mlp.up_proj",
                # "model.layers.31.mlp.down_proj"

                # t5
                # "decoder.block.11.layer.2.DenseReluDense.wi",
                # "decoder.block.11.layer.2.DenseReluDense.wo",
                # "decoder.block.11.layer.1.dropout"

                #     roberta
                # "encoder.layer.11.intermediate.dense",
                # "encoder.layer.11.output.dense"
            ],
            modules_to_save=[
            ],
            r=16,  # Rank of the low-rank matrices (r)
            lora_alpha=4,  # Expansion factor for the LoRA parameters (alpha)
            lora_dropout=0.1  # Dropout rate for LoRA adaptations
        )

        # Inject LoRA adaptations into the model
        self.model_lora = get_peft_model(self.model, config).to(self.device)

        # Enable training only for LoRA parameters
        for name, param in self.model_lora.named_parameters():
            if 'adapter' in name or 'lora' in name:
                param.requires_grad = True

        # Additional layers for downstream task-specific adaptations
        self.layer_norm = nn.LayerNorm(self.hidden).to(self.device)
        # self.mlp = nn.Linear(self.model.config.vocab_size, hidden_size).to(self.device)
        self.classifier = nn.Linear(self.hidden, 1).to(self.device)
        self.tanh = nn.Tanh()

    def print_trainable_layers(self):
        """Prints out the names of trainable layers within the model."""
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"Trainable: {name}")

    def forward(self, input_ids, attention_mask, generate_text=False):
        """Forward pass for generating text or classifying inputs."""
        input_ids = input_ids.long()
        attention_mask = attention_mask.long()

        # Obtain outputs from the model with LoRA layers
        lora_output = self.model_lora(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = lora_output[0]

        # Normalize and transform the last hidden state to get logits
        normalized_output = self.layer_norm(last_hidden_state)
        mean_last_token_logits = torch.mean(normalized_output, dim=1)
        tanh_output = self.tanh(mean_last_token_logits)

        # Generate classification output
        classification_output = self.classifier(tanh_output)

        return classification_output, mean_last_token_logits

# Example usage:
# model = ChatGLM_TEA(model_name='chatglm3-6b', adapter_size=64, hidden_size=65024)
# model.print_trainable_layers()