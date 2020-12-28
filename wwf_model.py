import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from words_with_friends import Game, get_vocab
import tqdm

device = torch.device("cuda")

training_iterations = 100

class SelfAttention(nn.Module):

    def __init__(self, in_sz, out_sz):
        super(SelfAttention, self).__init__()

        self.in_sz = in_sz
        self.out_sz = out_sz

        self.Q = nn.Linear(self.in_sz,self.out_sz)
        self.K = nn.Linear(self.in_sz,self.out_sz)
        self.V = nn.Linear(self.in_sz,self.out_sz)

    def forward(self, input):

        queries = self.Q(input)
        keys = self.K(input)
        values = self.V(input)

        scores = F.softmax(queries @ keys.T, dim = -1)
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

    def forward(self, input):

        tokenized = torch.tensor([self.vocab[x] for x in input]).to(device)
        emb = self.letter_emb(tokenized)
        positions = torch.tensor(np.arange(len(input))).to(device)
        emb_w_pos = emb + self.positional(positions)
        attn = self.word_SA(emb_w_pos)
        return attn.sum(dim = 0)

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

        return row + col + dir

class MoveEncoder(nn.Module):

    def __init__(self,vocab, in_sz, move_max):
        super(MoveEncoder, self).__init__()

        self.word_enc_sz = in_sz
        self.vector_enc_sz = in_sz
        self.score_enc_size = in_sz

        self.word_max = 15
        self.score_max = 100
        self.move_max = move_max

        self.word_enc = WordEncoder(self.word_max, vocab, self.word_enc_sz)
        self.vector_enc = VectorEncoder(self.word_max, self.vector_enc_sz)
        self.score_enc = nn.Embedding(100,self.score_enc_size)

    def forward(self,moves):

        encodings = []
        for move in moves[-self.move_max:]:
            word = self.word_enc(move.word)
            ordered = self.word_enc(move.word)

            vector = self.vector_enc(move)
            score = self.score_enc(torch.tensor([
                min(move.score,self.score_max - 1)
            ]).to(device)).squeeze()
            enc = word + ordered #+ vector + score
            encodings.append(enc)
        return torch.stack(encodings)

class Model(nn.Module):

    def __init__(self, vocab, move_max=10):
        super(Model, self).__init__()

        self.move_emb_size = 100
        self.hidden_sz = 100
        self.move_max = move_max

        self.network = nn.Sequential(
            MoveEncoder(vocab, self.move_emb_size, self.move_max),
            nn.LeakyReLU(),
            SelfAttention(self.move_emb_size, self.hidden_sz),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_sz,1),
            nn.Flatten(0),
            nn.Softmax(dim = -1)
        )

    def forward(self,states,possibles):

        policy = []
        for moves in possibles:
            probs = self.network(moves)
            policy.append(probs)
        return policy

    def loss(self,policy,actions,disc_rewards):

        probabilities = torch.stack([policy[i][a] for i,a in enumerate(actions)])

        return (-probabilities.log() * disc_rewards).sum()

def discount(rewards, discount_factor=.99):
    rewards = np.array(rewards)
    rewards = np.tanh(rewards / 100)
    
    prev = 0
    discounted_rewards = np.copy(rewards).astype(np.float32)
    for i in range(1, len(discounted_rewards) + 1):
        discounted_rewards[-i] += prev * discount_factor
        prev = discounted_rewards[-i]
    return torch.tensor(discounted_rewards).to(device)

def generate_trajectory(model):
    states = []
    possibles = []
    a_indices = []
    rewards = []

    game = Game()
    state = game.reset()
    done = False
    while not done:
        possible = game.actions()[-model.move_max:]
        if len(possible) > 0:
            states.append(state)

            probs = model(states,[possible])[0].detach().cpu().numpy()
            a_idx = np.random.choice(len(probs),p=probs)
            action = possible[a_idx]

            state, reward, done = game.step(action)

            possibles.append(possible)
            a_indices.append(a_idx)
            rewards.append(reward)
        else:
            _, _, done = game.step(None)
    

    return states,possibles,a_indices,rewards

def train(model):
    outcome = []
    for i in tqdm.tqdm(range(training_iterations)):
        model.eval()
        states, possibles, actions, rewards = generate_trajectory(model)
        outcome += [rewards[-1] > 0.]
        disc_rewards = discount(rewards)
        """
        model.train()
        policy = model(states,possibles)
        loss = model.loss(policy,actions,disc_rewards)

        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()
        """
        print(f'{i+1}: Reward {rewards[-1]}')

def main():
    model = Model(get_vocab(),2).to(device)
    model.optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4, eps = 1e-4)

    train(model)

if __name__ == "__main__":
    main()
