from itertools import islice
from collections import Counter
import numpy as np
import random

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


class DataPipline:
    r_filepath=''
    cont_win=0
    voc_to_int={}
    int_to_voc={}

    def __init__(self,r_file_path:str, limit:int)
        self.r_file_path=r_file_path
        
        self.limit


    def run(self,cont_win:int,subsamping_thr:float) -> np.array:
        self.cont_win=cont_win
        words, word_count=read_and_process('data/short_corpus.txt')
        self.words = subsample_words(words, word_count, subsampling_thr)
        self.voc_to_int, self.int_to_voc=create_lookup_tables(self.words)
        data=

        # [print(f'{key}:{value}') for key,value in voc_to_int.items()]



    def read_and_process(file_path :str):
        ''' Processing file to list of words, all in lower case and interpunction as tags '''
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
        return words, word_count
                
    def get_drop_prob(x:float, threshold_value:float):
        ''' Equation of probability of word being dropped
            - x : frequency of word in text
            - threshold_value : defines how probable it is for word to be dropped
        '''
        return 1 - np.sqrt(threshold_value/x)

    def subsample_words(words:list, word_counts:dict, threshold_value:float):
        '''Subsampling of high frequency words so dataset can be more balanced
            - words: list of words as they appear in text
            - word_counts : dict of how many wrds appear each time
            - threshold value : float on which depends probablity of word being subsampled
             '''
        total_count = len(words)
        freq_words = {word: (word_counts[word]/total_count) for word in set(words)}
        subsampled_words = [word for word in words if random.random() < (1 - get_drop_prob(freq_words[word], threshold_value))]
        return subsampled_words


    def create_lookup_tables(words:list):
        '''
        Create lookup tables for vocabulary
        - words: list of words
        - return: 2 dictionaries, vocab_to_int, int_to_vocab
        '''
        word_counts = Counter(words)
        sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True) # descending freq order
        int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
        vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

        return vocab_to_int, int_to_vocab


    def true_batch(words:list,cont_win_size:int, voc_to_int:dict,int_to_voc):
        ''' 
        Creating dataset of pairs of target_words and their neighbors
        '''
        inter_list=[value for key, value in interpunction_map.items()]
        keys, _ = int_to_voc.items()
        nr_target_words=100
        target_words=[]
        while len(target_words) < nr_target_words:
            rand=random.randint(0,keys)
            if not int_to_voc[rand] in target_words and not int_to_voc[rand] in inter_list:
                target_words.append(int_to_voc[rand])

        neighbors=get_target_word_nei(word,words,cont_win_size)
        data = [[batch[0],batch[1][i]] for batch in neighbors for i in range(len(batch[1])) if batch[0] != batch[1][i]]
        return data
    
    


    def get_target_word_nei(word:str,words:list,cont_win:int)->list:
        ''' 
        Finds neighbors of words in the text
        - word: target word that neigbors we look to find
        - list of words in text
        '''
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
            neighbors.append([word,nei])
            ind+=1
        return neighbors
