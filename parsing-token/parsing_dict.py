from itertools import islice
from collections import Counter
import numpy as np

interpunction_map = {
    '.':  ' <PERIOD> ',
    ',':  ' <COMMA> ',
    '"':  ' <QUOTATION_MARK> ',
    ';':  ' <SEMICOLON> ',
    '!':  ' <EXCLAMATION_MARK> ',
    '?':  ' <QUESTION_MARK> ',
    '(':  ' <LEFT_PAREN> ',
    ')':  ' <RIGHT_PAREN> ',
    '--': ' <HYPHENS> ',
    ':':  ' <COLON> ',
}

def read_and_process(file_path :str):
    with open(file_path, 'r') as file:
        text=file.read()
    file.close()
    text= text.rstrip()
    text=text.lower()
    for key, value in interpunction_map.items():
        text = text.replace(key, value)
    words = text.split()

    word_count=Counter(words)
    # trimmed_words=[word for word in words if word_count(word)>=2]
    # return trimmed_words
    return words
                
def create_short_file(r_file_path :str, limit :int, w_file_path:str):
    with open(r_file_path, 'r') as file:
        lines=[line for line in islice(file,limit)]
        content=''.join(line for line in lines)
    file.close()

    with open(w_file_path, 'w') as f:
        f.write(content)
    f.close

def create_lookup_tables(words):
    """
    Create lookup tables for vocabulary
    - words: list of words
    - return: 2 dictionaries, vocab_to_int, int_to_vocab
    """
    word_counts = Counter(words)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True) # descending freq order
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

    return vocab_to_int, int_to_vocab

words=read_and_process('data/short_corpus.txt')
print(words)
voc_to_int, int_to_voc=create_lookup_tables(words)

# [print(f'{key}:{value}') for key,value in voc_to_int.items()]

def true_batch(words:list,cont_win_size:int, voc_to_int:dict):
    pass

def target_word(word:str,words:list,cont_win:int)->list:
    word_place=[i for i,x in enumerate(words) if x==word]
    neighbors=[]
    it=len(word_place)
    ind=0
    while  ind<it:
        if word_place[ind]< cont_win:
            start=0
        else:
            start=word_place[ind]-cont_win
        if len(words)-word_place[ind]<cont_win:
            end=len(words)-1
        else:   
            end=word_place[ind]+1+cont_win


        nei=[words[indx] for indx in range(start,end)]
        if '<PERIOD>' in nei:
            p_ind=nei.index('<PERIOD>')
            if p_ind+start>word_place[ind]:
                nei=nei[:p_ind]
            elif p_ind+start<word_place[ind]:
                nei=nei[p_ind:] 
        neighbors.append(nei)
        ind+=1
    return neighbors

cont_win_nei=target_word('the',words,4)
[print(nei) for nei in cont_win_nei]




