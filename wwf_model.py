import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from words_with_friends import Game, get_vocab
import tqdm
import matplotlib.pyplot as plt
import copy

device= torch.device("cuda")

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

    def forward(self, input):

        tokenized = torch.tensor([self.vocab[x] for x in input]).to(device)
        emb = self.letter_emb(tokenized)
        positions = torch.tensor(np.arange(len(input))).to(device)
        emb_w_pos = emb + self.positional(positions)
        attn = self.word_SA(emb_w_pos).sum(dim = 0)
        return attn.tanh()

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
    
class MoveEncoder(nn.Module):

    def __init__(self, vocab, out_sz, word_max = 15):
        super(MoveEncoder, self).__init__()

        self.word_enc_sz = self.hidden_sz
        self.vector_enc_sz = self.hidden_sz
        concat_sz = self.word_enc_sz * 2 + self.vector_enc_sz

        self.word_max = word_max

        self.word_enc = WordEncoder(self.word_max, vocab, self.word_enc_sz)
        self.vector_enc = VectorEncoder(self.word_max, self.vector_enc_sz)
        self.score_enc = nn.Linear(1,concat_sz)

        hdn0 = 200
        hdn1 = 150
        
        self.encoder = Sequential(
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
        
        # Add score bias   
        normalized = torch.tensor(scores).to(device).float().view(-1,1)
        normalized -= normalized.min()
        if normalized.max() != 0:
            normalized /= normalized.max()
        scores_enc = self.score_enc(normalized)
        moves_enc = move_enc + scores_enc

        # Variational AE
        enc = self.encoder(moves_enc)
        enc_mu = self.encoder_mu(enc)
        enc_log_var = self.encoder_log_var(enc)

        epsilon = torch.rand(enc_mu.shape)
        return enc_mu + (enc_log_var / 2).exp() * epsilon

class MoveDecoder(nn.Module):

    def __init__(self, vocab, out_sz):
        super(MoveDecoder, self).__init__()
    
class BoardEncoder(nn.Module):

    def __init__(self,vocab, out_sz):
        super(BoardEncoder, self).__init__()

        self.hidden_sz = 50
        
        self.vocab = vocab
        self.vocab_sz = len(vocab)

        self.letter_emb = nn.Embedding(self.vocab_sz,self.hidden_sz)
        self.board_enc = nn.Sequential(
            nn.Conv2d(self.hidden_sz,self.hidden_sz,3,2,1),
            nn.LeakyReLU(),
            nn.Conv2d(self.hidden_sz,self.hidden_sz,3,2,1),
            nn.LeakyReLU(),
            nn.Conv2d(self.hidden_sz,self.hidden_sz,3,2,1),
            nn.Flatten(0),
            nn.Linear(self.hidden_sz * 4, out_sz)
        )

    def forward(self, input):

        tokenized = torch.zeros(np.shape(input.board)).long().to(device)
        for i in range(len(input.board)):
            for j in range(len(input.board[i])):
                tokenized[i,j] = self.vocab[input.board[i,j]]
        tokenized = tokenized
        board_emb = self.letter_emb(tokenized).permute(2,0,1).unsqueeze(0)
        
        board_enc = self.board_enc(board_emb)
        return board_enc
    
class StateEncoder(nn.Module):

    def __init__(self, vocab, out_sz, move_max):
        super(StateEncoder, self).__init__()

        self.hidden_sz = 100

        self.board_enc_sz = self.hidden_sz

        self.board_enc = BoardEncoder(vocab, self.hidden_sz)
        self.states_attn = SelfAttention(self.hidden_sz, self.hidden_sz, True)

    def forward(self, states):

        states_embs = []
        for state in states:
            board_enc = self.board_enc(state)

            states_embs.append(board_enc)
        states_embs = torch.stack(states_embs)
        states_attn = self.states_attn(states_embs)[-1]
        return states_attn
    
class Model(nn.Module):

    def __init__(self, vocab, move_max=10):
        super(Model, self).__init__()

        self.move_emb_sz = 100
        self.state_emb_sz = 100
        self.hidden_sz = self.move_emb_sz# + self.state_emb_sz
        self.move_max = move_max

        self.move_encoder = nn.Sequential(
            MoveEncoder(vocab, self.move_emb_sz),
            nn.GroupNorm(1,self.move_emb_sz)
        )
        #self.state_encoder = StateEncoder(vocab,self.state_emb_sz,self.move_max) 
        self.network = nn.Sequential(
            #SelfAttention(self.move_emb_sz, self.move_emb_sz),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_sz,1),
            nn.Flatten(0),
            nn.Softmax(dim = -1)
        )

    def forward(self,states,possibles):

        policy = []
        for moves,state in zip(possibles,states):
            moves_enc = self.move_encoder(moves)
            #state_enc = self.state_encoder(state).repeat(moves_enc.shape[0],1)
            #enc = torch.cat((moves_enc,state_enc),dim=-1)
            probs = self.network(moves_enc)
            policy.append(probs)
        return policy

    def loss(self,policy,actions,disc_rewards):

        probabilities = torch.stack([policy[i][a] for i,a in enumerate(actions)])

        return (-probabilities.log() * disc_rewards).sum()

class AutoEncoder(nn.Module):

    def __init__(self, vocab):

        self.move_emb_sz = 100
        self.move_encoder = MoveEncoder(vocab, self.move_emb_sz)
        self.move_decoder = MoveDecoder(vocab, self.move_emb_sz)
    
def discount(rewards_info, discount_factor = .6):
    game_won = rewards_info[-1].main_player_winning()

    rewards = torch.zeros(len(rewards_info)).float()
    for i in range(len(rewards_info)):
        rewards[i] = rewards_info[i].get_reward(game_won)

    discounted = torch.zeros(rewards.shape)
    discounted[-1] = rewards[-1]
    for i in range(len(rewards)-1):
        discounted[-(i+2)] = rewards[-(i+1)] + discounted[-(i+1)] \
            * discount_factor
    print(game_won)
    print(discounted)
    return discounted.to(device)

def generate_trajectory(model):
    states = []
    possibles = []
    a_indices = []
    rewards = []

    game = Game(seed=np.random.randint(max_games))
    state = game.reset()
    done = False

    #acs = [1,0,0,0,0]

    i = 0
    avg_prob = 0.
    while not done:
        possible = game.actions(model.move_max)
        if len(possible) == 0:
            _,_,done = game.step(None,None)
            continue

        states.append(state)
        
        probs = model([state],[possible])[0].detach().cpu().numpy()    
        a_idx = np.random.choice(len(possible),p=probs)
        avg_prob += probs[a_idx]
        
        state, reward, done = game.step(possible,a_idx)
        
        possibles.append(possible)
        a_indices.append(a_idx)
        rewards.append(reward)

        p_scores = [p.score for p in game.players]
        #print(np.round(probs,3),"\t",p_scores)
        #print(a_idx,end="")
        #print("\t",probs)
        i+=1
        if i == max_recursion:
            break
    scores = [p.score for p in game.players]
    #print(round(avg_prob/i,2),scores)

    return states,possibles,a_indices,rewards,scores[0] - max(scores[1:])

def all_possibilities(game=None,recursion="",seed=0):
    print(recursion)

    if recursion == "":
        game = Game(seed=seed)
        game.reset()

    possibilities = game.actions(max_moves)
    if len(possibilities) == 0 or len(recursion) == max_recursion:
        return {recursion:game.main_player_value()}

    value_dict = {}
    for i in range(len(possibilities)):
        game_copy = copy.deepcopy(game)
        game_copy.step(possibilities,i)
        value_dict_sub = all_possibilities(game_copy,recursion+str(i))
        for key in value_dict_sub:
            value_dict[key] = value_dict_sub[key]
    return value_dict

def test(model,seed=None):
    model.eval()
    game = Game(seed=seed)
    state = game.reset()
    done = False
    rewards = []

    i = 0
    while not done:
        possible = game.actions(model.move_max)
        if len(possible) > 0:
            probs = model([state],[possible])[0].detach().cpu().numpy()
            a_idx = np.argmax(probs)

            state, reward, done = game.step(possible, a_idx)
            rewards.append(reward)
            
            p_scores = [p.score for p in game.players]
            m_scores = [m.score for m in possible]
            print(np.round(probs,3),"\t",m_scores,"\t",p_scores)
        else:
            _, _, done = game.step(None,None)
        i+=1
        if i ==	max_recursion:
            break
    discount(rewards)
    
    scores = [p.score for p in game.players]
    return scores[0] - max(scores[1:])

def train(model):
    states, possibles, actions, rewards, diff = generate_trajectory(model)
    disc_rewards = discount(rewards)
    
    model.train()
    policy = model(states,possibles)
    loss = model.loss(policy,actions,disc_rewards)

    model.optimizer.zero_grad()
    loss.backward()
    model.optimizer.step()
    return diff

def main():
    global max_recursion
    max_recursion = np.PINF
    global max_moves
    max_moves = 2
    global max_games
    max_games = 5
    
    model = Model(get_vocab(),max_moves).to(device)
    model.optimizer = torch.optim.Adam(model.parameters(), lr = 5e-4, eps = 1e-4)
    """
    all_possible = all_possibilities(seed=2)
    print(all_possible)
    maxi = (None,np.NINF)
    for key in all_possible:
        if all_possible[key] > maxi[1]:
            maxi = (key,all_possible[key])
    print(maxi)
    """
    #output = []
    training_iterations = 200
    testing_iterations = 10
    iter_test = 10
    
    for i in tqdm.tqdm(range(training_iterations)):
        train(model)
        
        #sc = test(model)
        #print(sc)
        #output.append(sc)
    torch.save(model,"model.pt")
        
    #model = torch.load("model.pt")
    print(test(model,2))
    #for i in range(max_games):
    #    print(test(model,i))

    #print(output)
    #plt.plot(np.arange(len(output)),output)
    #plt.savefig("figure.png")

if __name__ == "__main__":
    main()
