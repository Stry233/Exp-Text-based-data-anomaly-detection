from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelWithLMHead
from base.base_net import BaseNet
import torch.nn as nn
import torch


# 1. 算法在新领域（已有）的应用
#    -》 算法不怎么变，check exitsing method 的 adaptation
#    -》 试图变成一个existing问题（GPT ZERO, PLAGIARISM DETECTION）
#    -》 看paper： check 数据集 availability
#    -》 潜在问题： embedding space不好，if we have an as good as GPT, which is more focus on content, not style

# 1. !!!!FIX A TOPIC!!!!
#    --> E.G. GPT DETECTION: 分解成小问题：对style 单独编码（research review）
#    -->
#    --> detect a single case... (找问题，找应用)

# 2. e.g. 倘若我有一个threshold来manually set, 我就可以用一个自动的算法来替代
#    --> NN uncertanty/probability
#    ---> RL: DROP OUT OUTLIER 后再进行average 会不会更好
#    --》 ASR:

# 3. reinforcement learning idea?

# 4. 改一改：找到
class text_Transformer(BaseNet):
    def __init__(self, hidden_size=512, model='gpt2'):
        super(text_Transformer, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        self.encoder = AutoModelForCausalLM.from_pretrained(model)
        self.encoder.resize_token_embeddings(len(self.tokenizer))

        self.rep_dim = hidden_size
        self.fc = nn.Linear(self.encoder.config.hidden_size, hidden_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, input_text):
        # inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        inputs = input_text
        outputs = self.encoder(**inputs, output_hidden_states=True)
        # Get the logits
        logits = outputs.hidden_states[-1]
        # Create a summary representation by averaging across the sequence dimension
        avg_rep = logits.mean(dim=1)
        return self.fc(avg_rep)


class text_transformer_decoder(BaseNet):
    def __init__(self, model='gpt2'):
        super(text_transformer_decoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        self.fc1 = nn.Linear(512, 768)

        self.decoder = AutoModelForCausalLM.from_pretrained(model)
        self.decoder.resize_token_embeddings(len(self.tokenizer))

        self.fc2 = nn.Linear(50258, 25)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, encoded):
        # Assuming encoded is of shape [batch_size, hidden_size], repeat it to match the sequence length
        #print(self.fc1(encoded).shape)
        decoder_outputs = self.decoder(inputs_embeds=self.fc1(encoded), attention_mask=None)
        #print(self.fc2(decoder_outputs.logits))
        return self.fc2(decoder_outputs.logits)


class text_Transformer_Autoencoder(BaseNet):
    def __init__(self):
        super(text_Transformer_Autoencoder, self).__init__()
        self.encoder = text_Transformer()
        self.decoder = text_transformer_decoder()

    def forward(self, input_text):
        encoded = self.encoder(input_text)
        decoded = self.decoder(encoded)
        return decoded
