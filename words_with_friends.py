from pynput.mouse import Listener,Controller
import pyscreenshot as ImageGrab
from PIL import Image
import os
import itertools
import re
import copy

def imager(fc,sc,var):
    if var == '-':
        a = 490
        b = 1962
    elif var == '+':
        a = 478
        b = 1955
    return ImageGrab.grab(bbox=(fc._position[0],fc._position[1], sc._position[0], sc._position[1])).resize((1242,2208)).crop((0,a,1242,b))

def on_click(x, y, button, pressed):
    if not pressed:
        return False

def set_position(clicker):
    with Listener(on_click=on_click) as listener:
        listener.join()
        clicker._position = mouse.position

class clicker:
    def __init__(self):
        self._position = None
        set_position(self)

class board_getter:

    def __init__(self,im,var):
        self._board_info = self.text_returner(im,var)

    def instantiate_array(self):
        A = ['TL',None,'TW',None,None,None,None,None,'TW',None,'TL']
        B = [None,'DW',None,None,None,'DW',None,None,None,'DW',None]
        C = ['TW',None,'TL',None,'DL',None,'DL',None,'TL',None,'TW']
        D = [None,None,None,'TL',None,None,None,'TL',None,None,None]
        E = [None,None,'DL',None,None,None,None,None,'DL',None,None]
        F = [None,'DW',None,None,None,None,None,None,None,'DW',None]
        G = [None,None,'DL',None,None,None,None,None,'DL',None,None]
        H = [None,None,None,'TL',None,None,None,'TL',None,None,None]
        I = ['TW',None,'TL',None,'DL',None,'DL',None,'TL',None,'TW']
        J = [None,'DW',None,None,None,'DW',None,None,None,'DW',None]
        K = ['TL',None,'TW',None,None,None,None,None,'TW',None,'TL']
        return [A,B,C,D,E,F,G,H,I,J,K]

    def pasty(self,image,x):
        img = Image.new('RGB', (x,x), (255, 255, 255))
        img.paste(image,(0,img.size[1]-image.size[1]))
        return img

    def letter_comparison(self,tile,letter):
        x = 200
        t = self.pasty(tile.resize((tile.size[0]*3,tile.size[1]*2)),x)
        l = self.pasty(letter.resize((tile.size[0]*3,tile.size[1]*2)),x)
        tpx = t.load()
        lpx = l.load()
        diff = 0
        for i in range(x):
            for j in range(x):
                diff += abs(tpx[i,j][0]-lpx[i,j][0])
        t_diff = diff / (x**2) / 255
        return t_diff

    def row_column(self,tile):
        pixels = tile.load()
        ic = 0
        summ = 0
        while ic < tile.size[1] and summ is 0:
            for j in range(tile.size[0]):
                summ += 255-pixels[j,ic][0]
            if summ is 0:
                ic += 1
        jc = 0
        summ = 0
        while jc < tile.size[0] and summ is 0:
            for i in range(tile.size[1]):
                summ += 255-pixels[jc,i][0]
            if summ is 0:
                jc += 1
        i_c = tile.size[1]-1
        summ = 0
        while i_c > ic and summ is 0:
            for j in range(1,tile.size[0]):
                summ += 255-pixels[tile.size[0]-j,i_c][0]
            if summ is 0:
                i_c -= 1
        j_c = tile.size[0]-1
        summ = 0
        while j_c > jc and summ is 0:
            for i in range(1,tile.size[1]):
                summ += (255-pixels[j_c,tile.size[1]-i][0])
            if summ is 0:
                j_c -= 1
        return[jc,ic,j_c,i_c]

    def red_box(self,tile):
        pixels = tile.load()
        s1 = 27
        s2 = 18
        x = tile.size[0]
        y = tile.size[1]
        red = tile.crop((x-s1,y-s2,x,y))
        rpixels = red.load()
        summr = 0
        summ = 0
        for r in range(s2):
            for c in range(s1):
                summr += rpixels[c,r][0]
                summ += rpixels[c,r][1]+rpixels[c,r][2]
        rindex = summr / (s1*s2)
        gbindex = summ / (s1*s2)
        if rindex > 220 and gbindex < 215:
            for r in range(s2):
                for c in range(s1):
                    pixels[c+x-s1,r+y-s2] = (150,150,150)
        return tile

    def orange_tile(self,tile):
        pixels = tile.load()
        for i in range(tile.size[0]):
            for j in range(tile.size[1]):
                summ = pixels[i,j][0]+pixels[i,j][1]+pixels[i,j][2]
                if summ > 600 or summ < 300:
                    pixels[i,j] = (0,0,0)
                else:
                    pixels[i,j] = (255,255,255)
        cr_index = self.row_column(tile)
        if cr_index[0] is tile.size[0] and cr_index[1] is tile.size[1]:
            return '*'
        cropped = tile.crop((cr_index[0],cr_index[1],cr_index[2],cr_index[3]))
        path = '/Users/blinnbryce/Documents/2018/pythonCode/tiles/'
        m = 1
        o = None
        for file in os.listdir(path):
            if file != '.DS_Store':
                letter = Image.open(path+file)
                x = self.letter_comparison(cropped,letter)
                if x < m:
                    m = x
                    o = file.split('.')[0]
        return o

    def text_returner(self,im,var):
        grid = self.instantiate_array()
        oranges = 0
        length = 11
        for c in range(length):
            for r in range(length):
                s = im.size[0]/length
                tile = im.crop((s*c+15,s*r+38,s*(c+1)-15,s*(r+1)-15))
                pixels = tile.load()
                bsumm = 0
                rsumm = 0
                for i in range(tile.size[0]):
                    for j in range(tile.size[1]):
                        bsumm += pixels[i,j][2]
                        rsumm += pixels[i,j][0]
                b = bsumm / (tile.size[0]*tile.size[1])
                re = rsumm / (tile.size[0]*tile.size[1])
                #print(str(r)+" "+str(c)+" "+str(re)+" "+str(b))
                if (b < 100  and re > 236) or (b > 100 and re > 230) or b < 70:
                    letter = self.orange_tile(self.red_box(tile))
                    grid[r][c] = letter
                    if letter != 'TW':
                        oranges += 1
        return [grid,oranges,self.options_finder(im,var)]

    def blank_tile(self,option):
        pixels = option.load()
        bsumm = 0
        for i in range(option.size[0]):
            for j in range(option.size[1]):
                bsumm += pixels[i,j][2]
        b = bsumm / (option.size[0]*option.size[1])
        return b < 110

    def options_finder(self,im,var):
        length = 7
        options = []
        for c in range(length):
            option = im.crop((25+177*c,im.size[1]-88,150+177*c,im.size[1]))
            if self.blank_tile(option):
                options.append(self.orange_tile(option))
        return options

class game:
    def __init__(self):
        self._words = {}
        self._alphabet = {}
        self._grid = []
        self._options = []
        self._available = []
        self._dictionary = self.make_dictionary()
        self._alphabet = {}
        self._available = []
        self._possible = []

    def update(self,board_info):
        self._grid = board_info[0]
        self._options = board_info[2]
        for row in self._grid:
            print(row)
        print('options =')
        print(self._options)
        print()
        if board_info[1] is not 0:
            self.update_words(self._grid)
            self._possible = self.possible_words()
            self.possible_placements(self._possible)
        else:
            self.empty_board()
        if len(self._available) is 0:
            print('There are no available options found')
        else:
            self.alphabeta()
            self.scoring()
        self.reset()

    def star(self,combo):
        possible = []
        combos = []
        place = ord('A')
        for i in range(26):
            letter = chr(place+i)
            combo[0] = letter
            words = self.in_dictionary(sorted(combo),0,len(self._dictionary))
            if words is not None:
                for word in words.split(','):
                    for x in range(len(word)):
                        if word[x] is letter:
                            temp = word[:x] + '*' + word[x+1:]
                            possible.append([word,temp,letter])
        return possible

    def empty_board(self):
        middle = int(len(self._grid)/2+0.5)-1
        combos = self.options_combos(self._options)
        for combo in combos:
            if len(combo) > 1:
                if '*' in combo:
                    possible = self.star(combo)
                    for i in possible:
                        self._available.append([[middle,middle],[i[1],i[2]],0,len(i[0]),True])
                else:
                    words = self.in_dictionary(combo,0,len(self._dictionary))
                    if words is not None:
                        for word in words.split(','):
                            self._available.append([[middle,middle],[word,None],0,len(word),True])

    def options_combos(self,letters):
        combos = []
        for L in range(7):
            for subset in itertools.combinations(letters,L+1):
                if sorted(subset) not in combos:
                    combos.append(sorted(subset))
        return combos

    def reset(self):
        self._words = {}
        self._grid = []
        self._options = []
        self._available = []
        self._possible = []

    def additional_score(self,tile,index,line):
        repls = ['TW','DW','DL','TL',None]
        wordx = 1
        if line[index] in repls:
            if line[index] == 'TW':
                wordx *= 3
            elif line[index] == 'DW':
                wordx *= 2
        score = 0
        a = index-1
        b = index+1
        isNone = False
        while isNone is False and a >= 0:
            if line[a] in repls:
                isNone = True
            else:
                score += self._alphabet[line[a]]
            a -= 1
        isNone = False
        while isNone is False and b < 11:
            if line[b] in repls:
                isNone = True
            else:
                score += self._alphabet[line[b]]
            b += 1
        if score is not 0:
            score += self._alphabet[tile]
        return score*wordx

    def scoring(self):
        scores = []
        repls = ['TL','TW','DL','DW']
        for i in self._available:
            index = i[0]
            word = i[1][0]
            letter = i[1][1]
            rc = i[2]
            added = i[3]
            isEmpty = i[4]
            extra = 0
            if added is 7:
                index[1-rc] -= len(word)-6
                extra = 35
            wordx = 1
            y = index[1-rc]
            score = 0
            adds = 0
            for z in range(len(word)):
                if rc is 0:
                    tile = self._grid[index[rc]][z+y]
                else:
                    tile = self._grid[z+y][index[rc]]
                if tile is None or tile in repls:
                    line = self.get_row_column(self._grid,z+y,1-rc)
                    adds += self.additional_score(word[z],index[rc],line)
                letx = 1
                if tile in repls:
                    if tile == 'DL':
                        letx *= 2
                    elif tile == 'DW':
                        wordx *= 2
                    elif tile == 'TL':
                        letx *= 3
                    elif tile == 'TW':
                        wordx *= 3
                score += letx*self._alphabet[word[z]]
            score *= wordx
            score += extra + adds
            to_print = ''.join(word)
            if '*' in word:
                splits = to_print.split('*')
                to_print = splits[0]+letter+splits[1]
            scores.append([score,to_print,index,rc])
        scores = sorted(scores, reverse=True)
        for i in range(len(scores)):
            index = scores[i][2]
            if scores[i][3] is 0:
                align = 'row'
            else:
                align = 'column'
            print('Option #'+str(i+1)+' is: '+scores[i][1]+', a '+align+' word at ['+str(index[0])+', '+str(index[1])+'] which gives '+str(scores[i][0])+' points')
        print()

    def alphabeta(self):
        points = [1,4,4,2,1,4,3,3,1,10,5,2,4,2,1,4,10,1,1,1,2,5,4,8,3,10]
        place = ord('A')
        for i in range(26):
            self._alphabet[chr(place+i)] = points[i]
        self._alphabet['*'] = 0

    def get_row_column(self,grid,place,rc):
        if rc is 0: #row word
            line = grid[place]
        else: #column word
            line = [row[place] for row in grid]
        return line

    def possible_placements(self,possible):
        can_place = []
        repls = ['TL','TW','DL','DW',None]
        for i in possible:
            gindex = i[0]
            subword = i[1]
            word = i[2]
            windex = i[3]
            index = gindex[:2]
            rc = gindex[2]
            line = self.get_row_column(copy.deepcopy(self._grid),index[rc],rc)
            for x in range(len(line)):
                if line[x] in repls:
                    line[x] = ''
            start = index[1-rc]-windex
            if self.fits(line,start,word,subword):
                if self.mutate(word,start,line,gindex,rc,repls):
                    if rc is 0:
                        place = [index[0],start]
                    else:
                        place = [start,index[1]]
                    if i[4] is not None:
                        to_add = i[4]
                    else:
                        to_add = [i[2],None]
                    self._available.append([place,to_add,rc,len(word)-len(subword),False])

    def mutate(self,word,start,line,gindex,rc,repls):
        grid = copy.deepcopy(self._grid)
        for spot in range(len(word)):
            line[spot+start] = word[spot]
        if rc is 0:
            grid[gindex[rc]] = line
        else:
            for i in range(len(line)):
                grid[i][gindex[rc]] = line[i]
        yes = True
        s = start
        while yes is True and s < start+len(word):
            line2 = self.get_row_column(grid,s,1-rc)
            for x in range(len(line2)):
                if line2[x] in repls or line2[x] is None:
                    line2[x] = ' '
            mash = ''.join(line2).split()
            for i in mash:
                if len(i) > 1:
                    mashed_word = self.in_dictionary(sorted(i),0,len(self._dictionary))
                    if mashed_word is None or i not in mashed_word.split(','):
                        yes = False
            s += 1
        return yes

    def fits(self,line,start,word,subword):
        a = start-1
        b = start+len(word)
        if start >= 0:
            if start-1 >= 0:
                if line[start-1] != '':
                    return False
        else:
            return False
        if start+len(word)-1 < len(line):
            if start+len(word) < len(line):
                if line[start+len(word)] != '':
                    return False
        else:
            return False
        e_length = len(''.join(line))+len(word)-len(subword)
        linecopy = copy.deepcopy(line)
        linecopy[start:b] = word
        r_length = len(''.join(linecopy))
        if e_length != r_length:
            return False
        for i in range(len(line)):
            if line[i] != '' and line[i] != linecopy[i]:
                return False
        return True

    def make_dictionary(self):
        dictionary = []
        text_file = open('dict.txt')
        for line in text_file.readlines():
            dictionary.append(line.strip().split('-'))
        text_file.close()
        return dictionary

    def in_dictionary(self,word,a,b):
        if a == b:
            if a == len(self._dictionary) or len(word) != len(self._dictionary[a][0]):
                return None
            for i in range(len(word)):
                if word[i] != self._dictionary[a][0][i]:
                    return None
            return self._dictionary[a][1]
        if b-a is 1:
            a1 = self.in_dictionary(word,a,a)
            b1 = None
            if b is not len(self._dictionary):
                b1 = self.in_dictionary(word,b,b)
            if a1 is not None or b1 is not None:
                if a1 is not None:
                    return a1
                return b1
            return None
        loc = int((a+b)/2)
        wloc = list(self._dictionary[loc][0])
        s = 0
        while s < min(len(word),len(wloc))-1 and word[s] is wloc[s]:
            s += 1
        v = ord(wloc[s])-ord(word[s])
        if v is 0:
            if len(word) > len(wloc):
                return self.in_dictionary(word,loc,b)
            elif len(wloc) > len(word):
                return self.in_dictionary(word,a,loc)
            return self._dictionary[loc][1]
        if v > 0:
            return self.in_dictionary(word,a,loc)
        return self.in_dictionary(word,loc,b)

    def possible_words(self):
        combos = self.options_combos(self._options)
        possible = []
        store = {}
        for i in combos:
            for j in self._words:
                short = j.replace("-","")
                k=sorted(list(short)+i)
                p = len(j.replace("-"," ").split())
                if p == 1 or len(short)+len(i) == len(j):
                    if '*' in k:
                        truths = self.star(k)
                    else:
                        truths = [[self.in_dictionary(k,0,len(self._dictionary))]]
                    for z in truths:
                        truth = z[0]
                        if len(z) > 1:
                            stars = z[1:3]
                        else:
                            stars = None
                        if truth is not None:
                            found_words = truth.split(',')
                            for found in found_words:
                                locations = self.multifind(found,j)
                                for loc in locations:
                                    for index in self._words[j]:
                                        to_add = [[index[:2]+[index[3]],list(j),found,loc,stars]]
                                        if len(i) is 1 and p is 1 and index[2] <= 1:
                                            to_add.extend(self.sideways(i[0],j,index,found,stars,loc))
                                        self.filter(store,to_add,possible)

        return possible

    def filter(self,store,to_add,possible):
        for i in to_add:
            if i[2] in store:
                if i[0] not in store[i[2]]:
                    possible.append(i)
                    store[i[2]].append(i[0])
            else:
                possible.append(i)
                store[i[2]] = [i]


    def sideways(self,letter,b_word,index,found,stars,loc):
        combos = []
        for L in range(1,7):
            for subset in itertools.combinations(self._options,L+1):
                sort = sorted(subset)
                if letter in sort:
                    words = self.in_dictionary(sort,0,len(self._dictionary))
                    if words is not None:
                        for word in words.split(','):
                            s_indices = self.sideways_index(index,b_word,word,letter,found)
                            if s_indices is not None:
                                for s_index in s_indices:
                                    combos.append([s_index,[],word,0,stars])
        return combos

    def sideways_index(self,index,b_word,word,letter,found):
        if index[2] == 0:
            l = len(b_word)*(1-found.find(b_word))-found.find(b_word)
            if index[3] is 0:
                shift = index[1] + l
                i = [index[0],shift]
            else:
                shift = index[0] + l
                i = [shift,index[1]]
            if shift >= 11 or shift < 0:
                return None
            spots = self.multifind(word,letter)
        else:
            if index[3] is 0:
                i = [index[0],index[1]+b_word.find('-')]
            else:
                i = [index[0]+b_word.find('-'),index[1]]
            spots = self.multifind(word,letter)
        s_indices = []
        for spot in spots:
            if index[3] is 0:
                shift2 = i[0]-spot
                s_index = [shift2,i[1],1]
            else:
                shift2 = i[1]-spot
                s_index = [i[0],shift2,0]
            if shift2 + len(word) - 1 >= 11 or shift2 < 0:
                return None
            s_indices.append(s_index)
        return s_indices

    def multifind(self,string,value):
        starts = []
        values = []
        for i in range(len(string)):
            if value[0] == string[i]:
                starts.append(i)
        for start in starts:
            sub = string[start:]
            if len(sub) >= len(value):
                valid = True
                j = 0
                while j < len(value) and valid:
                    if value[j] != sub[j] and value[j] != '-':
                        valid = False
                    j += 1
                if valid:
                    values.append(start)
        return values

    def adding(self,p,rc,index):
        x = p[0]
        got = self._words.get(x)
        if rc is 0:
            coord = [index,p[2][1]]
        else:
            coord = [p[2][1],index]
        a = coord+[p[1]]+[rc]
        if got is not None:
            got.append(a)
            self._words[x] = got
        else:
            self._words[x] = [a]

    def get_empty(self,ws):
        if len(ws) is 1:
            return [0,ws[0][0]]
        word = ws[0][0]
        empty = 0
        for i in range(0,len(ws)-1):
            x = ws[i][1]+len(ws[i][0])
            y = ws[i+1][1]
            empty += (y-x)
            word += '-'*(y-x)+ws[i+1][0]
        return [empty,word]

    def combinations(self,ws):
        words = []
        b = 0
        while b <= len(ws)-1:
            for c in range(b,len(ws)):
                info = self.get_empty(ws[b:c+1])
                empty = info[0]
                new = info[1]
                if empty <= 7:
                    words.append([new,empty,ws[b]])
            b += 1
        return words

    def update_words(self,grid):
        repls = ['TL','TW','DL','DW',None]
        for r in range(len(self._grid)):
            word = ''
            words = []
            for i in range(len(grid[r])):
                c = grid[r][i]
                if c not in repls:
                    word += c
                else:
                    if len(word) is not 0:
                        words.append([word,i-len(word)])
                        word = ''
            if len(word) is not 0:
                words.append([word,i-len(word)+1])
            if len(words) is not 0:
                pop = self.combinations(words)
                for p in pop:
                    self.adding(p,0,r)
        for c in range(len(self._grid)):
            col = [row[c] for row in grid]
            word = ''
            words = []
            for i in range(len(col)):
                r = col[i]
                if r not in repls:
                    word += r
                else:
                    if len(word) is not 0:
                        words.append([word,i-len(word)])
                        word = ''
            if len(word) is not 0:
                words.append([word,i-len(word)+1])
            if len(words) is not 0:
                pop = self.combinations(words)
                for p in pop:
                    self.adding(p,1,c)

if __name__ == "__main__":
    mouse = Controller()
    print('do a first clicc')
    fc = clicker()
    print('do a second clicc')
    sc = clicker()
    game = game()
    """var = None
    while var != '+' and var != '-':
        var = input('mode (+ or -): ')
    ready = input('u ready? ')
    while ready == '':
        if ready == 'again':
            print('do a first clicc')
            fc = clicker()
            print('do a second clicc')
            sc = clicker()
        else:
            print()
            im = imager(fc,sc,var)
            bg = board_getter(im,var)
            game.update(bg._board_info)
            ready = input('u ready? ')"""

    var = '+'
    im = imager(fc,sc,var)
    bg = board_getter(im,var)
    game.update(bg._board_info)
