import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from words_with_friends import Game, get_vocab
import tqdm
import matplotlib.pyplot as plt
import copy

device= torch.device("cuda")
PAD = "PAD"

class SelfAttention(nn.Module):

    def __init__(self, in_sz, out_sz, use_mask = False):
        super(SelfAttention, self).__init__()

        self.in_sz = in_sz
        self.out_sz = out_sz
        self.use_mask = use_mask

        self.Q = nn.Linear(self.in_sz,self.out_sz)
        self.K = nn.Linear(self.in_sz,self.out_sz)
        self.V = nn.Linear(self.in_sz,self.out_sz)

    def forward(self, input):

        queries = self.Q(input)
        keys = self.K(input)
        values = self.V(input)

        raw = queries @ keys.T
        if self.use_mask:
            ninf = np.ones(raw.shape) * np.NINF
            tril = torch.tensor(np.triu(ninf,1)).to(device)
        scores = F.softmax(raw, dim = -1)
        return scores @ values

class WordEncoder(nn.Module):

    def __init__(self, length, vocab, in_sz):
        super(WordEncoder, self).__init__()

        self.length = length
        self.in_sz = in_sz
        self.vocab = vocab
        self.vocab_sz = len(vocab)

        self.letter_emb = nn.Embedding(self.vocab_sz,self.in_sz)
        self.positional = nn.Embedding(self.length,self.in_sz)
        self.word_SA = SelfAttention(self.in_sz, self.in_sz)

    def forward(self, word):

        tokenized = torch.tensor([self.vocab[x] for x in word]).to(device)
        emb = self.letter_emb(tokenized)
        positions = torch.tensor(np.arange(len(word))).to(device)
        emb_w_pos = emb + self.positional(positions)
        attn = self.word_SA(emb_w_pos).sum(dim = 0)
        return attn.tanh()

class WordDecoder(nn.Module):

    def __init__(self, length, vocab, in_sz):
        super(WordDecoder, self).__init__()

        self.length = length
        self.in_sz = in_sz
        self.vocab_sz = len(vocab)

        self.word_SA = SelfAttention(self.in_sz, self.in_sz)
        self.word_network = nn.Sequential(
            nn.Linear(self.in_sz,hdn0),
            nn.LeakyReLU(),
            nn.Linear(hdn0,hdn1),
            nn.LeakyReLU(),
            nn.Linear(hdn1,self.vocab_sz),
            nn.Softmax(dim=-1)
        )

    def forward(self, input):

        attn = self.word_SA(input)
        word_pred = self.word_network(attn)
        return word_pred

class VectorEncoder(nn.Module):

    def __init__(self, length, in_sz):
        super(VectorEncoder, self).__init__()

        self.row_emb = nn.Embedding(length,in_sz)
        self.col_emb = nn.Embedding(length,in_sz)
        self.dir_emb = nn.Embedding(2,in_sz)

    def forward(self,move):

        row = self.row_emb(torch.tensor([move.row]).to(device)).squeeze()
        col = self.col_emb(torch.tensor([move.column]).to(device)).squeeze()
        dir = self.dir_emb(torch.tensor([move.dir]).to(device)).squeeze()

        enc = row + col + dir
        return enc.tanh()

class VectorDecoder(nn.Module):

    def __init__(self, length, in_sz):
        super(VectorDecoder, self).__init__()

        self.row_dec = nn.Embedding(in_sz,length)
        self.col_dec = nn.Embedding(in_sz,length)
        self.dir_dec = nn.Embedding(in_sz,2)

    def forward(self,input):

        row = self.row_emb(input)
        col = self.col_emb(input)
        dir = self.dir_emb(input)

        return row, col, dir

class MoveEncoder(nn.Module):

    def __init__(self, vocab, out_sz, word_max = 15):
        super(MoveEncoder, self).__init__()

        self.hidden_sz = 100
        self.word_enc_sz = self.hidden_sz
        self.vector_enc_sz = self.hidden_sz
        concat_sz = self.word_enc_sz * 2 + self.vector_enc_sz

        self.word_max = word_max

        self.word_enc = WordEncoder(self.word_max, vocab, self.word_enc_sz)
        self.vector_enc = VectorEncoder(self.word_max, self.vector_enc_sz)
        self.score_enc = nn.Linear(1,concat_sz)

        hdn0 = 200
        hdn1 = 150

        self.downsampler = Sequential(
            nn.Linear(concat_sz,hdn0),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(hdn0,hdn1),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(hdn1,out_sz)
        )

        self.encoder_mu = nn.Linear(out_sz,out_sz)
        self.encoder_log_var = nn.Linear(out_sz,out_sz)

    def forward(self,moves):
        # Encode move
        moves_enc = []
        scores = []
        for move in moves:
            word = self.word_enc(move.word)
            ordered = self.word_enc(move.word)
            vector = self.vector_enc(move)

            move_enc = torch.cat((word,ordered,vector),dim=-1)
            moves_enc.append(move_enc)
            scores.append(move.score)
        moves_enc = torch.stack(moves_enc)

        # Encode score
        normalized = torch.tensor(scores).to(device).float().view(-1,1)
        normalized -= normalized.min()
        if normalized.max() != 0:
            normalized /= normalized.max()
        score_bias = self.score_enc(normalized)

        # Add score bias
        move_enc = move_enc + score_bias

        # Downsample
        downsampled = self.downsampler(move_enc)
        # Variational
        enc_mu = self.encoder_mu(downsampled)
        enc_log_var = self.encoder_log_var(downsampled)

        epsilon = torch.rand(enc_mu.shape)
        encoded = enc_mu + (enc_log_var / 2).exp() * epsilon
        return encoded, enc_mu, enc_log_var

class MoveDecoder(nn.Module):

    def __init__(self, vocab, in_sz, word_max = 15):
        super(MoveDecoder, self).__init__()

        self.hidden_sz = 100
        self.word_enc_sz = self.hidden_sz
        self.vector_enc_sz = self.hidden_sz
        concat_sz = self.word_enc_sz * 2 + self.vector_enc_sz

        self.word_max = word_max

        hdn0 = 150
        hdn1 = 200

        self.upsampler = Sequential(
            nn.Linear(in_sz,hdn0),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(hdn0,hdn1),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(hdn1,in_sz)
        )

        self.score_separate = nn.Linear(in_sz,in_sz)
        self.score_dec = nn.Linaer(in_sz,1)

        self.word_dec = WordDecoder(self.word_max, vocab, self.word_enc_sz)
        self.vec_dec = VectorDecoder(self.word_max, self.vector_enc_sz)

        hdn0 = 100
        hdn1 = 50

        self.length_dec = nn.Sequential(
            nn.Linear(self.word_enc_sz * 2,hdn0),
            nn.LeakyReLU(),
            nn.Linear(hdn0,hdn1),
            nn.LeakyReLU(),
            nn.Linear(hdn1,self.word_max),
            nn.Softmax(dim=-1)
        )

    def forward(self, encoded, return_moves=False):
        # Upsample
        upsampled = self.upsampler(encoded)

        # Remove score bias
        score_bias = self.separate(upsampled)
        move_enc = upsampled - score_bias

        # Decode score
        score_raw = self.score_dec(score_bias)

        # Decode move
        word_enc, ordered_enc, vector_enc = move_enc.reshape(-1,3,self.hidden_sz).permute(1,0,2)
        length_raw = self.length_dec(torch.cat((word_enc,ordered_enc),dim=-1))
        word_raw = self.word_dec(word_enc)
        ordered_raw = self.word_dec(ordered_enc)
        vector_raw = self.vec_dec(vector_enc)

        if not return_move:
            return length_raw, word_raw, ordered_raw, vector_raw, score_raw

        # word_probs,word_len_probs = word_raw
        # word_len = word_len_probs.clone().detach().cpu().argmax(dim=-1)
        # word_letters = word_probs.clone().detach().cpu().argmax(dim=-1)

class VarAutoEncoder(nn.Module):

    def __init__(self, vocab, word_max=15):

        self.vocab = vocab
        self.move_emb_sz = 100
        self.word_max = word_max
        self.move_encoder = MoveEncoder(vocab, self.move_emb_sz, word_max)
        self.move_decoder = MoveDecoder(vocab, self.move_emb_sz, word_max)
        self.optimizer = torch.optim.Adam(self.parameters(), lr = 1e-3, eps = 1e-4)

    def forward(self,moves):

        encoding, enc_mu, enc_log_var = self.move_encoder(moves)
        decoded = self.move_decoder(encoding)
        return decoded, enc_mu, enc_log_var

    def loss(decoded, labels, enc_mu, enc_log_var):
        cel = nn.CrossEntropyLoss(reduction='none')

        len_probs, word_probs, ordered_probs, vec_probs, score_pred = decoded

        len_labels, masks, word_labels, ordered_labels, \
            vec_labels, score_labels = labels

        loss = 0

        # Length prediction loss
        loss += cel(len_probs,len_labels).mean()

        # Word reconstruction loss
        loss += (cel(word_probs,word_labels) * masks).mean()

        # Ordered reconstruction loss
        loss += (cel(ordered_probs,ordered_labels) * masks).mean()

        # Vector reconstruction loss
        row_probs, col_probs, dim_probs = vector_probs
        loss += cel(row_probs,vec_labels[:,0])
        loss += cel(col_probs,vec_labels[:,1])
        loss += cel(dim_probs,vec_labels[:,2])

        # Score prediction loss
        loss += (score_pred - score_labels) ** 2.

        # KL loss
        loss += 0.5 * (1 + enc_log_var - enc_mu ** 2. - enc_log_var.exp())

        return loss

def generate_moves(model):
    all_moves = []

    game = Game()
    game.reset()
    done = False

    while not done:
        moves = game.actions()
        all_moves += moves

        if len(possible) == 0:
            _,_,done = game.step(None,None)
            continue

        a_idx = np.random.choice(len(moves))
        _, _, done = game.step(moves,a_idx)

    return all_moves

def get_labels(batch,vocab,word_max):
    len_labels = []
    masks = []
    word_token_labels = []
    ordered_token_labels = []
    vec_labels = []
    score_labels = []
    for move in batch:
        length = len(move.word)
        len_labels.append(length)

        mask = [1.] * length + (move_max - length) * [0.]
        masks.append(mask)

        word_tokenized = [vocab[letter] for letter in move.word] + \
            vocab[PAD] * (move_max - length)
        word_token_labels.append(word_tokenized)

        ordered_tokenized = [vocab[letter] for letter in move.ordered_tiles] + \
            vocab[PAD] * (move_max - length)
        ordered_token_labels.append(ordered_tokenized)

        vec_labels.append([move.row,move.col,move.dir])

        score_labels.append(move.score)
    len_labels = torch.tensor(len_labels).to(device)
    masks = torch.tensor(masks).to(device)
    word_token_labels = torch.tensor(word_token_labels).to(device)
    ordered_token_labels = torch.tensor(ordered_token_labels).to(device)
    vec_labels = torch.tensor(vec_labels).to(device)
    score_labels = torch.tensor(score_labels).to(device)
    return len_labels, masks, word_token_labels, ordered_token_labels, vec_labels, score_labels

def train(vae):
    moves = generate_moves()
    batch_sz = 10
    for i in range(0,len(moves),batch_sz):
        batch = moves[i:i+bsz]
        labels = get_labels(batch,vae.vocab,vae.word_max)
        decoded, enc_mu, enc_log_var = vae(batch)
        loss = vae.loss(decoded, labels, enc_mu, enc_lov_var)

        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()

def main():

    vae = VarAutoEncoder(get_vocab())

    for i in range(100):
        train(vae)


if __name__ == "__main__":
    main()
