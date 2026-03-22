from itertools import islice
from collections import Counter
import numpy as np
import random

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

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
    voc_size=0

    def __init__(self,r_file_path:str):
        self.r_file_path=r_file_path


    def run(self,cont_win:int,subsampling_thr:float,nr_target_words:int,neg_sampl:int) -> np.array:
        self.cont_win=cont_win
        words, word_count=self.read_and_process()
        self.words = self.subsample_words(words, word_count, subsampling_thr)
        self.voc_to_int, self.int_to_voc=self.create_lookup_tables()
        self.voc_size=len(self.voc_to_int)
        data=self.true_batch(nr_target_words,neg_sampl)
        return data


    def read_and_process(self):
        ''' Processing file to list of words, all in lower case and interpunction as tags '''
        with open(self.r_file_path, 'r') as file:
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
                
    def get_drop_prob(self,x:float, threshold_value:float):
        ''' Equation of probability of word being dropped
            - x : frequency of word in text
            - threshold_value : defines how probable it is for word to be dropped
        '''
        return 1 - np.sqrt(threshold_value/x)

    def subsample_words(self,words:list, word_counts:dict, threshold_value:float):
        '''Subsampling of high frequency words so dataset can be more balanced
            - words: list of words as they appear in text
            - word_counts : dict of how many wrds appear each time
            - threshold value : float on which depends probablity of word being subsampled
             '''
        total_count = len(words)
        freq_words = {word: (word_counts[word]/total_count) for word in set(words)}
        subsampled_words = [word for word in words if random.random() < (1 - self.get_drop_prob(freq_words[word], threshold_value))]
        return subsampled_words


    def create_lookup_tables(self):
        '''
        Create lookup tables for vocabulary
        - return: 2 dictionaries, vocab_to_int, int_to_vocab
        '''
        word_counts = Counter(self.words)
        sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True) # descending freq order
        int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
        vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

        return vocab_to_int, int_to_vocab


    def true_batch(self,nr_target_words:int,neg_sampl:int):
        ''' 
        Creating dataset of pairs of target_words and their neighbors
        '''
        inter_list=[value for key, value in interpunction_map.items()]
        keys, values = zip(*self.int_to_voc.items())
        del values
        target_words=[]

        while len(target_words) < nr_target_words:
            rand=random.randint(0,len(keys))
            if not self.int_to_voc.get(rand) in target_words and not self.int_to_voc.get(rand) in inter_list:
                target_words.append(self.int_to_voc.get(rand))

        neighbors=[]
        for word in target_words:
            neighbors.extend(self.get_target_word_nei(word))

        data = [[batch[0],batch[1][i]] for batch in neighbors for i in range(len(batch[1])) if batch[0] != batch[1][i]]
        neg= self.negative_sampling(neighbors,neg_sampl)
        data = [[self.voc_to_int.get(pair[0]), self.voc_to_int.get(pair[1])] for pair in data]
        neg = [[self.voc_to_int.get(pair[0]), self.voc_to_int.get(pair[1])] for pair in neg]

        return data,neg
    
    def negative_sampling(self,neighbors:list, neg_samples:int):
        neg=[]
        for batch in neighbors:
            i=0
            while i<neg_samples:
                rand=self.int_to_voc.get(random.randint(0,len(self.int_to_voc)))
                if not rand in batch[1] and rand!=batch[0]:
                    neg.append([batch[0],rand,0])
                    i=i+1
        return neg




    def get_target_word_nei(self,word:str)->list:
        ''' 
        Finds neighbors of words in the text
        - word: target word that neigbors we look to find
        - list of words in text
        '''
        word_place=[i for i,x in enumerate(self.words) if x==word]
        neighbors=[]
        it=len(word_place)
        ind=0
        while  ind<it:
            if word_place[ind]< self.cont_win:
                start=0
            else:
                start=word_place[ind]-self.cont_win
            if len(self.words)-word_place[ind]<=self.cont_win:
                end=len(self.words)-1
            else:   
                end=word_place[ind]+1+self.cont_win

            nei=[self.words[indx] for indx in range(start,end)]
            if '<PERIOD>' in nei:
                p_ind=nei.index('<PERIOD>')
                if p_ind+start>word_place[ind]:
                    nei=nei[:p_ind]
                elif p_ind+start<word_place[ind]:
                    nei=nei[p_ind:] 
            neighbors.append([word,nei])
            ind=ind+1
        return neighbors
    

