import torch.nn as nn
import torch

class LSTM_Decoder(nn.Module):
  def __init__(self, vocab_size, embed_dim, hidden_size, padding_idx, device) -> None:
    super(LSTM_Decoder, self).__init__()
    
    self.device = device
    self.special_token = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}
    self.vocab_size = vocab_size
    self.padding_idx = padding_idx
    self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx)
    self.lstm_cell = nn.LSTMCell(embed_dim, hidden_size) # LSTMのセル単位で処理を行う
    self.fc = nn.Linear(hidden_size, vocab_size)
  
  def forward(self, inputs, encoder_state, generate_len):
    hidden, cell = encoder_state
    
    if self.training == True:
      output = torch.zeros(inputs.size(1) ,inputs.size(0), self.vocab_size)
      embedded_vector = self.embedding(inputs)
      permuted_vec = torch.permute(embedded_vector, (1, 0, 2))
      
      for i in range(permuted_vec.size(0)):
        hidden, cell = self.lstm_cell(permuted_vec[i], (hidden, cell))
        output[i] = self.fc(hidden)
      
      return torch.permute(output, (1, 0, 2)) 
    
    else:
      # output = torch.zeros(generate_len, self.vocab_size, device=self.device)
      output = []
      
      for i, (h, c) in enumerate(zip(hidden, cell)):
        sentence_out = []
        
        h_x = h.unsqueeze(dim=0)
        c_x = c.unsqueeze(dim=0)
        
        input_tmp = torch.tensor([self.special_token["<BOS>"]], device=self.device)
        input_tmp = self.embedding(input_tmp)
        
        for j in range(generate_len):
          h_x ,c_x =self.lstm_cell(input_tmp, (h_x, c_x))
          output_tmp = self.fc(h_x)
          output_tmp = torch.argmax(output_tmp)

          sentence_out.append(output_tmp.to("cpu").detach().numpy().copy().item())
          
          if output_tmp == self.special_token["<EOS>"]:
            break
          
          input_tmp = self.embedding(output_tmp).unsqueeze(dim=0)
        output.append(sentence_out)

      return output