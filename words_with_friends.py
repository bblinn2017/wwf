import numpy as np

import os
from itertools import combinations, combinations_with_replacement
import re
import copy

ROW = 0
COLUMN = 1
INVALID = {'TL','TW','DL','DW',None}
L_MULTI = {'DL':2,'TL':3}
W_MULTI = {'DW':2,'TW':3}
STAR = '*'
SPACE = "-"

class Player:
    def __init__(self,id):
        self.id = id
        self.tiles = []
        self.score = 0

    def __str__(self):
        return f'Player {self.id}, Tiles {self.tiles}, Score {self.score}'

    def add_tile(self,tile):
        self.tiles.append(tile)

    def remove_tile(self,tile):
        self.tiles.remove(tile)

    def pick_move(self,moves):
        return moves[-1]

class Board:
    def __init__(self):
        self.make_board()
        self.make_dictionary()
        self.make_tiles()
        self.stars = []
        self.game_continue = True

    # Initialization logic
    def make_board(self):

        layout = {
            'TW':[[0,0],[0,7],[0,14],[7,0],
                [7,14],[14,0],[14,7],[14,14]],
            'TL':[[1,5],[1,9],[5,1],[5,5],
                [5,9],[5,13],[9,1],[9,5],
                [9,9],[9,13],[13,5],[13,9]],
            'DW':[[1,1],[1,13],[2,2],[2,12],
                [3,3],[3,11],[4,4],[4,10],
                [7,7],[10,4],[10,10],[11,3],
                [11,11],[12,2],[12,12],[13,1],
                [13,13]],
            'DL':[[0,3],[0,11],[2,6],[2,8],
                [3,0],[3,7],[3,14],[6,2],
                [6,6],[6,8],[6,12],[7,3],
                [7,11],[8,2],[8,6],[8,8],
                [8,12],[11,0],[11,7],[11,14],
                [12,6],[12,8],[14,3],[14,11]],
        }

        self.board = \
        np.reshape([None]* 15 ** 2,(15,15))

        for key in layout:
            for r,c in layout[key]:
                self.board[r,c] = key

    def make_dictionary(self):
        dictionary = {}
        with open('Collins Scrabble Words (2019).txt','r') as f:
            for line in f:
                word = line.strip()
                key = ''.join(sorted(word))
                if key not in dictionary:
                    dictionary[key] = []
                dictionary[key].append(word)
        self.dictionary = dictionary

    def make_tiles(self):

        self.amount =  {
            'A':9,'B':2,'C':2,'D':4,
            'E':12,'F':2,'G':3,'H':2,
            'I':9,'J':1,'K':1,'L':4,
            'M':2,'N':6,'O':8,'P':2,
            'Q':1,'R':6,'S':4,'T':6,
            'U':4,'V':2,'W':2,'X':1,
            'Y':2,'Z':1,STAR:2
        }

        self.value = {
            'A':1,'B':3,'C':3,'D':2,
            'E':1,'F':4,'G':2,'H':4,
            'I':1,'J':8,'K':5,'L':1,
            'M':3,'N':1,'O':1,'P':3,
            'Q':10,'R':1,'S':1,'T':1,
            'U':1,'V':4,'W':4,'X':8,
            'Y':4,'Z':10,STAR:0
        }

    # Move finding logic
    class Group:
        def __init__(self,row,column,string,dir):

            self.row = row
            self.column = column
            self.string = string
            self.dir = dir

        def __str__(self):
            return f'Group: Row {self.row}, Col {self.column}, Direction along {"row" if self.dir == ROW else "col"}, String {self.string}'

    class MultiGroup:
        def __init__(self,groups,string,dir):

            self.row = groups[0].row
            self.column = groups[0].column
            self.string = string
            self.dir = dir

        def __str__(self):
            return f'MultiGroup: Row {self.row}, Col {self.column}, Direction along {"row" if self.dir == ROW else "col"}, String {self.string}'

    def get_line_groups(self,index,dir):

        line = self.board[index] if dir == ROW else self.board[:,index]
        line = ['' if x in INVALID else x for x in line]

        groups = []
        string = ''
        for i in range(len(line)):
            if line[i] != '':
                string += line[i]
            elif len(string) > 0:
                row = index if dir == ROW else i - len(string)
                column = index if dir == COLUMN else i - len(string)
                groups.append(self.Group(row,column,string,dir))
                string = ''
        if len(string) > 0:
            row = index if dir == ROW else i+1 - len(string)
            column = index if dir == COLUMN else i+1 - len(string)
            groups.append(self.Group(row,column,string,dir))

        def get_multi_string(candidates):

            string = candidates[0].string

            for i in range(len(candidates)-1):
                if dir == ROW:
                    end_first = candidates[i].column + len(candidates[i].string)
                    start_second = candidates[i+1].column
                else:
                    end_first = candidates[i].row + len(candidates[i].string)
                    start_second = candidates[i+1].row
                string += (start_second - end_first) * SPACE + candidates[i+1].string

            return string

        multi_groups = []
        for first_index in range(len(groups)):
            for second_index in range(first_index,len(groups)):
                candidate_groups = groups[first_index:second_index+1]
                multi_string = get_multi_string(candidate_groups)

                multi_groups.append(self.MultiGroup(
                    candidate_groups,multi_string,dir
                ))
        return multi_groups

    def get_board_tile_groups(self):

        tile_groups = []
        for r in range(np.shape(self.board)[0]):
            tile_groups += self.get_line_groups(r,ROW)
        for c in range(np.shape(self.board)[1]):
            tile_groups += self.get_line_groups(c,COLUMN)
        return tile_groups

    class Match:

        def __init__(self,word,string,tiles,star,mg=None,idx=None,r=None,c=None,dir=None):

            self.word = word
            self.string = string
            self.tiles = tiles
            self.star = star
            if mg is not None:
                self.row = mg.row - (idx if mg.dir != ROW else 0)
                self.column = mg.column - (idx if mg.dir != COLUMN else 0)
                self.dir = mg.dir
            else:
                self.row = r
                self.column = c
                self.dir = dir
            if len(tiles+star) == 0:
                print(self)
                raise Exception("yoion")

        def __str__(self):

            return f'Match {self.word}, Current {self.string}, Tiles {self.tiles}, Star {self.star}, at ({self.row},{self.column}), Direction along {"row" if self.dir == ROW else "col"}'

        def __eq__(self, obj):
            return str(self) == str(obj)

        def __hash__(self):
            return hash(str(self))

    def get_empty_matches(self,tiles,star=[]):

        if STAR in tiles:
            count = tiles.count(STAR)
            tiles_wo_star = list(filter(lambda a: a != STAR, tiles))

            alphabet = list(self.amount.keys())
            alphabet.remove(STAR)

            combos = []
            for i in range(count):
                combos += combinations_with_replacement(alphabet,i+1)

            star_matches = []
            for item in combos:
                star_matches += self.get_empty_matches(tiles_wo_star,list(item))
            return star_matches

        for c in self.get_tile_combinations(tiles+star):
            if len(c) == 1:
                continue

            sort_string = ''.join(sorted(c))
            if sort_string not in self.dictionary:
                continue

            matched_words = self.dictionary[sort_string]

            matches = []

            string = SPACE * len(sort_string)
            for word in matched_words:
                for i in range(len(word)):
                    start = 7 - (len(word) - 1) + i
                    matches.append(self.Match(
                        word,string,tiles,star,r=start,c=7,dir=COLUMN
                    ))
                    matches.append(self.Match(
                        word,string,tiles,star,r=7,c=start,dir=ROW
                    ))

        return matches

    def get_sideways_word_matches(self,tiles,mg,star,word):

        dir = mg.dir if len(mg.string) > 1 else 1 - mg.dir

        # word can be placed anywhwere such that it touches parallel word
        first = (mg.column if dir == ROW else mg.row) - (len(word) - 1)
        first = max(0,first)
        last = (mg.column if dir == ROW else mg.row) + (len(mg.string) - 1)
        last = min(len(self.board)-1,last)

        def get_parallel(line,displacement):
            curr_matches = []

            for i in range(first,last+1):
                start_idx = max(0,i-1)
                end_idx = min(len(self.board)-1,i+len(word))

                # slice cannot contain any letters, otherwise redundant match
                slice = line[start_idx:end_idx+1]

                if np.any([x not in INVALID for x in slice]):
                    continue

                string = SPACE * len(word)
                row = mg.row + displacement if dir == ROW else i
                col = mg.column + displacement if dir == COLUMN else i

                end = (row if dir != ROW else col) + len(word)
                if row < 0 or col < 0 or end > len(self.board):
                    continue

                curr_matches.append(
                    self.Match(word,string,tiles,star,r=row,c=col,dir=dir)
                )

            return curr_matches

        matches = []

        # check two parallel lines
        if (dir == ROW and mg.row > 0) or (dir == COLUMN and mg.column > 0):
            line = self.board[mg.row-1] if dir == ROW else self.board[:,mg.column-1]
            matches += get_parallel(line,-1)
        if (dir == ROW and mg.row < len(self.board)-1) or (dir == COLUMN and mg.column < len(self.board)-1):
            line = self.board[mg.row+1] if dir == ROW else self.board[:,mg.column+1]
            matches += get_parallel(line,1)

        return matches

    def get_sideways_matches(self,tiles,mg,star=[]):

        # a sideways match has to be longer than 1
        if len(tiles+star) == 1:
            return []

        # only consider single groups to avoid redundancy
        if SPACE in mg.string:
            return []

        if STAR in tiles:
            count = tiles.count(STAR)
            tiles_wo_star = list(filter(lambda a: a != STAR, tiles))

            alphabet = list(self.amount.keys())
            alphabet.remove(STAR)

            combos = []
            for i in range(count):
                combos += combinations_with_replacement(alphabet,i+1)

            star_matches = []
            for item in combos:
                star_matches += self.get_sideways_matches(tiles_wo_star,mg,list(item))
            return star_matches

        # a sideways match must be a word
        sort_string = ''.join(sorted(tiles+star))
        if sort_string not in self.dictionary:
            return []

        matched_words = self.dictionary[sort_string]
        matches = []
        for word in matched_words:
            matches += self.get_sideways_word_matches(tiles,mg,star,word)

        return matches

    def get_word_matches(self,tiles,mg,star,word):

        matches = []

        # get the locations where word could line up with multigroup string
        start_idx = [i for i,x in enumerate(word) if x == mg.string[0]]

        for idx in start_idx:

            # align multigroup string with word
            string = SPACE * idx + mg.string

            # string length must be less than word length
            # otherwise, this positioning does not fill in all the spaces
            if len(string) > len(word):
                continue

            # extend string with blanks if its shorter than word
            string = string + SPACE*(len(word) - len(string))

            # array for word and string
            string_arr = np.array(list(string))
            word_arr = np.array(list(word))

            # check if spaces match
            is_match = np.logical_or(string_arr == word_arr,
                string_arr == SPACE).all()

            if is_match:
                row = mg.row - (idx if mg.dir != ROW else 0)
                column = mg.column - (idx if mg.dir != COLUMN else 0)
                dir = mg.dir

                # check if fits on board
                start = column if mg.dir == ROW else row
                end = (row if dir != ROW else column) + len(word) - 1
                if row < 0 or column < 0 or end >= len(self.board):
                    continue

                # check if fits in line without intersecting another word
                line = self.board[row] if mg.dir == ROW else self.board[:,column]
                if start != 0 and line[start-1] not in INVALID:
                    continue
                if end != len(self.board)-1 and line[end+1] not in INVALID:
                    continue

                matches.append(self.Match(word,string,tiles,star,mg=mg,idx=idx))

        return matches

    def get_matches(self,tiles,mg,star=[]):

        string = list(mg.string)
        compiled = list(filter(lambda a: a != SPACE, string))
        sort = sorted(compiled+tiles+star)

        # tokens do not fill the empty space
        if len(tiles) < mg.string.count(SPACE):
            return []

        # check for star token, get all star matches
        if STAR in sort:
            count = sort.count(STAR)
            tiles_wo_star = list(filter(lambda a: a != STAR, tiles))

            alphabet = list(self.amount.keys())
            alphabet.remove(STAR)

            combos = []
            for i in range(count):
                combos += combinations_with_replacement(alphabet,i+1)

            star_matches = []
            for item in combos:
                star_matches += self.get_matches(tiles_wo_star,mg,list(item))
            return star_matches

        # check for dictionary matches
        sort_string = ''.join(sort)
        if sort_string not in self.dictionary:
            return []

        matched_words = self.dictionary[sort_string]

        matches = []
        for word in matched_words:
            matches += self.get_word_matches(tiles,mg,star,word)

        return matches

    class Move:

        def __init__(self,match,ordered_tiles,score):

            self.word = match.word
            self.ordered_tiles = ordered_tiles

            self.row = match.row
            self.column = match.column
            self.dir = match.dir

            self.score = score

        def __str__(self):

            return f'Move {self.word}, Tiles {self.ordered_tiles}, at ({self.row},{self.column}), Direction along {"row" if self.dir == ROW else "col"}, Score {self.score}'

        def __eq__(self,obj):
            return str(self) == str(obj)

        def __hash__(self):
            return hash(str(self))

        def __lt__(self,other):
            if self.score != other.score:
                return self.score < other.score
            return self.word < other.word

    def cross_word_score(self,cross_line,token,idx):

        start = idx
        while start > 0:
            if cross_line[start-1] in INVALID:
                break
            start -= 1
        end = idx
        while end < len(self.board) - 1:
            if cross_line[end+1] in INVALID:
                break
            end += 1

        # if there's just one letter, it's not a word
        if start == end:
            return 0

        score = 0
        word_multi = 1
        for i in range(start,end):
            if i == idx:
                letter_multi = 1
                if token in L_MULTI:
                    letter_multi = L_MULTI[token]
                elif token in W_MULTI:
                    word_multi = W_MULTI[token]
                score += self.value[token] * letter_multi
            else:
                score += self.value[cross_line[i]]
        score *= word_multi
        return score

    def place_stars(self,line,dir,row,col):
        for r,c in self.stars:
            if dir == ROW and row == r:
                line[c] = STAR
            elif dir == COLUMN and col == c:
                line[r] = STAR
        return line

    def get_score(self,match,ordering):

        total_score = 0

        # score along direction and along cross direction
        line = np.array(self.board[match.row] if match.dir == ROW else  self.board[:,match.column])
        line = self.place_stars(line,match.dir,match.row,match.column)

        word_multi = 1
        score = 0

        start = match.column if match.dir == ROW else match.row
        word = match.word
        for i in range(len(word)):
            # direction word
            letter_multi = 1

            token = line[i+start]
            if token in L_MULTI:
                letter_multi = L_MULTI[token]
            if token in W_MULTI:
                word_multi *= W_MULTI[token]
            if token in INVALID:
                token = word[i]
            score += self.value[token] * letter_multi

            # cross direction word
            # if no new token is placed in this line then continue
            if line[i+start] not in INVALID:
                continue

            row = match.row if match.dir == ROW else i+start
            col = match.column if match.dir == COLUMN else i+start
            dir = 1 - match.dir

            cross_line = np.array(self.board[i+start] if match.dir != ROW else  self.board[:,i+start])
            cross_line = self.place_stars(cross_line,dir,row,col)

            idx = row if match.dir == ROW else match.column
            total_score += self.cross_word_score(cross_line,word[i],idx)

        total_score += score * word_multi

        # 7 tiles played is a bingo
        if len(match.tiles + match.star) == 7:
            total_score += 50

        return total_score

    def get_valid_moves(self,match):

        cond =  match.word == "AT" and match.row == 6 and match.column == 14

        # check if word fits in cross lines
        for i in range(len(match.word)):
            if match.string[i] is not SPACE:
                continue

            # get the index of the line in the word direction
            idx = i + (match.row if match.dir != ROW else match.column)
            # get the index of the line in the word direction
            idx_cross = match.row if match.dir == ROW else match.column

            line_cpy = np.array(self.board[idx] if match.dir != ROW else self.board[:,idx])
            line_cpy[idx_cross] = match.word[i]

            clean_line = [" " if x in INVALID else x for x in line_cpy]
            split_line = ''.join(clean_line).strip().split()

            for item in split_line:
                if len(item) == 1:
                    continue

                sort_string = ''.join(sorted(item))
                if sort_string not in self.dictionary:
                    return []
                matched_words = self.dictionary[sort_string]
                if item not in matched_words:
                    return []


        # get the tiles which will be placed in order
        word = match.word
        string = match.string
        to_place = ''.join(SPACE if string[i] != SPACE else word[i] for i in range(len(word)))

        def replace_with_star(currs, curr_stars):
            if len(curr_stars) == 0:
                return currs

            item = curr_stars.pop(0)

            replaced = []
            for curr in currs:
                curr_replaced = []
                idx = np.reshape((np.array(list(curr)) == item).nonzero(),[-1])
                for i in idx:
                    new = curr[:i]+STAR+curr[i+1:]
                    curr_replaced.append(new)
                replaced += replace_with_star(curr_replaced,curr_stars)
            return replaced

        if match.star == []:
            orderings = [to_place]
        else:
            # check how many combinations are necessary
            orderings = replace_with_star([to_place],copy.deepcopy(match.star))

        # match is valid, get moves
        moves = []
        for ordering in orderings:
            score = self.get_score(match,ordering)
            moves.append(self.Move(match,ordering,score))

        return moves

    def get_tile_combinations(self,tiles):
        combos = []
        for L in range(7):
            for subset in combinations(tiles,L+1):
                if sorted(subset) not in combos:
                    combos.append(sorted(subset))
        return combos

    def get_all_moves(self,player_tiles):

        # get all player tile combinations
        combos = self.get_tile_combinations(player_tiles)

        # get all possible multi groups on the board
        multi_groups = self.get_board_tile_groups()

        # board empty, different protocol
        if len(multi_groups) == 0:
            matches = self.get_empty_matches(player_tiles)
        # board not empty, find all matches in dictionary
        else:
            matches = []
            for mg in multi_groups:
                for c in combos:
                    matches += self.get_matches(c,mg)

                    # get sideways matches
                    matches += self.get_sideways_matches(c,mg)

        # determine if matches fit
        moves = []
        for match in matches:
            moves += self.get_valid_moves(match)

        return sorted(set(moves))

    # Game playing logic
    def draw_tile(self):

        keys = list(self.amount.keys())
        count = list(self.amount.values())
        p = np.array(count) / np.sum(count)

        idx = np.random.choice(len(keys),p=p)
        self.amount[keys[idx]] -= 1

        return keys[idx]

    def play_move(self,move):
        row = move.row
        col = move.column

        for i in range(len(move.ordered_tiles)):
            char = move.ordered_tiles[i]

            if char == SPACE:
                char = move.word[i]
            if char == STAR:
                self.stars.append((row,col))
                char = move.word[i]

            self.board[row,col] = char

            row += 1 if move.dir != ROW else 0
            col += 1 if move.dir != COLUMN else 0

    def write_board(self,filename):
        string = ''
        _line = '_' * (len(self.board) * 3 + 1)
        string += _line + "\n"
        for row in self.board:
            string += "| "
            for column in row:
                val = " " if column in INVALID else column
                string += val + "| "
            string += "\n" + _line + "\n"

        with open(filename,"w") as f:
            f.write(string)

    def fill_tiles(self,player):
        while len(player.tiles) < 7 and np.sum(list(self.amount.values())) > 0:
            player.tiles.append(self.draw_tile())

    def take_tiles(self,player,tiles):
        for tile in tiles:
            player.remove_tile(tile)

class Game:

    def __init__(self):
        super().__init__()

if __name__ == "__main__":
    game = Game()
    players = [Player(0),Player(1)]
    curr = 0
    moveless = set()

    def game_valid():
        for row in np.concatenate((game.board,game.board.T),axis=0):
            clean_row = [' ' if x in INVALID else x for x in row]
            split_row = ''.join(clean_row).strip().split()
            for item in split_row:
                if len(item) == 1:
                    continue
                valid = ''.join(sorted(item)) in game.dictionary
                if not valid:
                    return False
        return True

    while game.game_continue:
        curr_player = players[curr]
        game.fill_tiles(curr_player)

        moves = game.get_all_moves(curr_player.tiles)
        if len(moves) == 0:
            moveless.add(curr)
            curr = (curr + 1) % len(players)
            if len(moveless) == len(players):
                break
            continue
        else:
            moveless = set()

        move = curr_player.pick_move(moves)
        game.play_move(move)
        curr_player.score += move.score

        if game_valid() == False:
            print(game.board)
            exit()

        tiles = list(filter(lambda x: x != SPACE, moves[-1].ordered_tiles))
        game.take_tiles(curr_player,tiles)
        curr = (curr + 1) % len(players)

    print(players[0])
    print(players[1])
    print(f'Remaining Tiles {np.concatenate([[key]*game.amount[key] for key in game.amount])}')

    game.write_board("board.txt")
