import numpy as np

class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """
        
        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set


    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """

        decoded_path = [''] * y_probs.shape[2]
        blank = 0
        path_prob = [1] * y_probs.shape[2]

        # TODO:
        # 1. Iterate over sequence length - len(y_probs[0])
        # 2. Iterate over symbol probabilities
        # 3. update path probability, by multiplying with the current max probability
        # 4. Select most probable symbol and append to decoded_path
        # 5. Compress sequence (Inside or outside the loop)
        symbol_b = ['+'] + self.symbol_set
        for i in range(y_probs.shape[1]):
            for j in range(y_probs.shape[2]):
                y = y_probs[:, i, j]
                path_prob[j] *= np.max(y)
                decoded_path[j] += symbol_b[np.argmax(y)]
        sequence_compressed = [''] * y_probs.shape[2]
        for i in range(y_probs.shape[2]):
            path_i = decoded_path[i]
            for j, symbol in enumerate(path_i):
                if symbol != '+':
                    if path_i[j-1] != path_i[j] or j == blank:
                        sequence_compressed[i] += symbol
        return sequence_compressed[blank], path_prob[blank]


class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """

        self.symbol_set = symbol_set
        self.beam_width = beam_width

    def decode(self, y_probs):
        """
        
        Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
			batch size for part 1 will remain 1, but if you plan to use your
			implementation for part 2 you need to incorporate batch_size

        Returns
        -------
        
        forward_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores

        """

        Forward_Path = []
        Merged_Path_Scores = []

        # iterate through batches
        for i in range(y_probs.shape[2]):
            
            score_blank = {}
            path_blank = set('')
            score_path = {}
            path_symbol = set()
            score_blank[''] = y_probs[:, 0, i][0] # score of blank
            for k in range(len(self.symbol_set)):
                path = self.symbol_set[k]
                score_path[path] = y_probs[:, 0, i][k + 1] # score of symbol
                path_symbol.add(path) # add newpath
            
            # iterate through time steps
            for j in range (1, y_probs.shape[1]):

                # first confine only to beam_width
                scores = [score_blank[path] for path in path_blank] + [score_path[path] for path in path_symbol] # all scores stored here
                threshold = sorted(scores, reverse=True)[self.beam_width - 1] if self.beam_width < len(scores) else min(scores) # get threshold
                score_blank1 = {}
                path_blank1 = set()
                for path in path_blank:
                    if score_blank[path] >= threshold:
                        score_blank1[path] = score_blank[path]
                        path_blank1.add(path) # add newpath
                score_path1 = {}
                path_symbol1 = set()
                for path in path_symbol:
                    if score_path[path] >= threshold:
                        score_path1[path] = score_path[path]
                        path_symbol1.add(path) # add newpath

                # second extend path with blank
                score_blank = {}
                path_blank = set()
                for path in path_blank1: # ends with blank
                    score_blank[path] = score_blank1[path] * y_probs[:, j, i][0]
                    path_blank.add(path) # add newpath
                for path in path_symbol1: # ends with symbol
                    score_blank.setdefault(path, 0)
                    score_blank[path] += score_path1[path] * y_probs[:, j, i][0] # increment
                    path_blank.add(path) # add newpath

                # third extend path with symbol
                score_path = {}
                path_symbol = set()
                for path in path_blank1: # ends with blank
                    for k in range(len(self.symbol_set)):
                        score_path[path + self.symbol_set[k]] = score_blank1[path] * y_probs[:, j, i][k + 1]
                        path_symbol.add(path + self.symbol_set[k]) # add newpath
                for path in path_symbol1: # ends with symbol
                    for k in range(len(self.symbol_set)):
                        new_path = path if self.symbol_set[k] == path[-1] else path + self.symbol_set[k]
                        score_path.setdefault(new_path, 0)
                        score_path[new_path] += score_path1[path] * y_probs[:, j, i][k + 1] # path merge if newpath already exists
                        path_symbol.add(new_path) # add newpath

            for path in path_blank:
                score_path.setdefault(path, 0)
                score_path[path] += score_blank[path]
            Merged_Path_Scores.append(score_path)
            Forward_Path.append(max(score_path, key=lambda k: score_path[k])) # Find the path with the best score

        if y_probs.shape[2] > 1:
            return Forward_Path, Merged_Path_Scores
        else:
            return Forward_Path[0], Merged_Path_Scores[0]