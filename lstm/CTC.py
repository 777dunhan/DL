import numpy as np


class CTC(object):

    def __init__(self, BLANK=0):
        """

        Initialize instance variables

        Argument(s)
        -----------

        BLANK (int, optional): blank label index. Default 0.

        """

        # No need to modify
        self.BLANK = BLANK


    def extend_target_with_blank(self, target):
        """Extend target sequence with blank.

        Input
        -----
        target: (np.array, dim = (target_len,))
                target output containing indexes of target phonemes
        ex: [1,4,4,7]

        Return
        ------
        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended target sequence with blanks
        ex: [0,1,0,4,0,4,0,7,0]

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections
        ex: [0,0,0,1,0,0,0,1,0]
        """

        extended_symbols = [self.BLANK]
        for symbol in target:
            extended_symbols.append(symbol)
            extended_symbols.append(self.BLANK)

        N = len(extended_symbols)

        # -------------------------------------------->
        # TODO
        target_len = len(target)
        skip_connect = [self.BLANK] * (2 * target_len + 1)
        i = 1
        j = 2
        while i < target_len:
            if j % 2 == 1 and target[i] != target[i - 1]: # and i != 0: (not necessary)
                skip_connect[j] = 1
            if j % 2 == 1:
                i += 1
            j += 1
        # <---------------------------------------------

        extended_symbols = np.array(extended_symbols).reshape((N,))
        skip_connect = np.array(skip_connect).reshape((N,))

        return extended_symbols, skip_connect


    def get_forward_probs(self, logits, extended_symbols, skip_connect):
        """Compute forward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(Symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t, qextSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probabilities

        """

        S, T = len(extended_symbols), len(logits)
        alpha = np.zeros(shape=(T, S))

        # -------------------------------------------->
        alpha[0, 0] = logits[0, extended_symbols[0]] # TODO: Intialize alpha[0][0]
        alpha[0, 1] = logits[0, extended_symbols[1]] # TODO: Intialize alpha[0][1]
        # TODO: Compute all values for alpha[t][sym] where 1 <= t < T and 1 <= sym < S (assuming zero-indexing)
        # IMP: Remember to check for skipConnect when calculating alpha
        for i in range(1, T):
            alpha[i, 0] = alpha[i-1, 0] * logits[i, extended_symbols[0]]
            for j in range(1, S):
                if skip_connect[j] == 0:
                    alpha[i, j] = alpha[i-1, j-1] + alpha[i-1, j]
                else:
                    alpha[i, j] = alpha[i-1, j-2] + alpha[i-1, j-1] + alpha[i-1, j]
                alpha[i, j] *= logits[i, extended_symbols[j]]
        # <---------------------------------------------

        return alpha


    def get_backward_probs(self, logits, extended_symbols, skip_connect):
        """Compute backward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t,extSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probabilities
    
        """
        
        S, T = len(extended_symbols), len(logits)
        beta = np.zeros(shape=(T, S))

        # -------------------------------------------->
        beta[T-1, S-2] = 1
        beta[T-1, S-1] = 1
        for i in reversed(range(T - 1)):
            beta[i, S-1] = beta[i+1, S-1] * logits[i+1, extended_symbols[S-1]]
            for j in reversed(range(S - 1)):
                if j < S-2 and skip_connect[j+2] == 1:
                    beta[i, j] += beta[i+1, j+2] * logits[i+1, extended_symbols[j+2]]
                beta[i,j] += beta[i+1, j+1] * logits[i+1, extended_symbols[j+1]] + beta[i+1, j] * logits[i+1, extended_symbols[j]]
        # <--------------------------------------------

        return beta


    def get_posterior_probs(self, alpha, beta):
        """Compute posterior probabilities.

        Input
        -----
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probability

        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probability

        Return
        ------
        gamma: (np.array, dim = (input_len, 2 * target_len + 1))
                posterior probability

        """

        [T, S] = alpha.shape
        gamma = np.zeros(shape=(T, S))
        sumgamma = np.zeros((T,))
        
        # -------------------------------------------->
        for i in range(T):
            for j in range(S):
                gamma[i, j] = alpha[i, j] * beta[i, j]
                sumgamma[i] += gamma[i, j]
            gamma[i, :] /= sumgamma[i]
        # <---------------------------------------------

        return gamma


class CTCLoss(object):

    def __init__(self, BLANK=0):
        """

        Initialize instance variables

        Argument(s)
        -----------
        BLANK (int, optional): blank label index. Default 0.
        
        """
        # -------------------------------------------->
        # No need to modify
        super(CTCLoss, self).__init__()

        self.BLANK = BLANK
        self.gammas = []
        self.ctc = CTC()
        # <---------------------------------------------

    def __call__(self, logits, target, input_lengths, target_lengths):

        # No need to modify
        return self.forward(logits, target, input_lengths, target_lengths)

    def forward(self, logits, target, input_lengths, target_lengths):
        """CTC loss forward

        Computes the CTC Loss by calculating forward, backward, and
        posterior proabilites, and then calculating the avg. loss between
        targets and predicted log probabilities

        Input
        -----
        logits [np.array, dim=(seq_length, batch_size, len(symbols)]:
            log probabilities (output sequence) from the RNN/GRU

        target [np.array, dim=(batch_size, padded_target_len)]:
            target sequences

        input_lengths [np.array, dim=(batch_size,)]:
            lengths of the inputs

        target_lengths [np.array, dim=(batch_size,)]:
            lengths of the target

        Returns
        -------
        loss [float]:
            avg. divergence between the posterior probability and the target

        """

        # No need to modify
        self.logits = logits
        self.target = target
        self.input_lengths = input_lengths
        self.target_lengths = target_lengths

        #####  IMP:
        #####  Output losses should be the mean loss over the batch

        # No need to modify
        B, _ = target.shape
        total_loss = np.zeros(B)
        self.extended_symbols = []

        for batch_itr in range(B):
            # -------------------------------------------->
            # Computing CTC Loss for single batch
            # Process:
            #     Truncate the target to target length
            #     Truncate the logits to input length
            #     Extend target sequence with blank
            #     Compute forward probabilities
            #     Compute backward probabilities
            #     Compute posteriors using total probability function
            #     Compute expected divergence for each batch and store it in totalLoss
            #     Take an average over all batches and return final result
            # <---------------------------------------------

            # -------------------------------------------->
            # TODO
            logits_t = logits[:input_lengths[batch_itr], batch_itr, :]
            extended_symbols, skip_connect = self.ctc.extend_target_with_blank(target[batch_itr, :target_lengths[batch_itr]])
            gamma = self.ctc.get_posterior_probs(
                        self.ctc.get_forward_probs(logits_t, extended_symbols, skip_connect), 
                        self.ctc.get_backward_probs(logits_t, extended_symbols, skip_connect)
                    )
            for i in range(gamma.shape[0]):
                for j in range(gamma.shape[1]):
                    total_loss[batch_itr] -= gamma[i, j] * np.log(logits_t[i, extended_symbols[j]])
            # <---------------------------------------------

        total_loss = np.sum(total_loss) / B

        return total_loss


    def backward(self):
        """

        CTC loss backard

        Calculate the gradients w.r.t the parameters and return the derivative 
        w.r.t the inputs, xt and ht, to the cell.

        Input
        -----
        logits [np.array, dim=(seqlength, batch_size, len(Symbols)]:
            log probabilities (output sequence) from the RNN/GRU

        target [np.array, dim=(batch_size, padded_target_len)]:
            target sequences

        input_lengths [np.array, dim=(batch_size,)]:
            lengths of the inputs

        target_lengths [np.array, dim=(batch_size,)]:
            lengths of the target

        Returns
        -------
        dY [np.array, dim=(seq_length, batch_size, len(extended_symbols))]:
            derivative of divergence w.r.t the input symbols at each time

        """

        # No need to modify
        T, B, C = self.logits.shape
        dY = np.full_like(self.logits, 0)

        for batch_itr in range(B):
            # -------------------------------------------->
            # Computing CTC Derivative for single batch
            # Process:
            #     Truncate the target to target length
            #     Truncate the logits to input length
            #     Extend target sequence with blank
            #     Compute derivative of divergence and store them in dY
            # <---------------------------------------------

            # -------------------------------------------->
            # TODO
            logits_t = self.logits[:self.input_lengths[batch_itr], batch_itr, :]
            extended_symbols, skip_connect = self.ctc.extend_target_with_blank(self.target[batch_itr, :self.target_lengths[batch_itr]])
            gamma = self.ctc.get_posterior_probs(
                        self.ctc.get_forward_probs(logits_t, extended_symbols, skip_connect), 
                        self.ctc.get_backward_probs(logits_t, extended_symbols, skip_connect)
                    )
            for i in range(gamma.shape[0]):
                for j in range(gamma.shape[1]):
                    dY[i, batch_itr, extended_symbols[j]] -= gamma[i, j] / logits_t[i, extended_symbols[j]]
            # <---------------------------------------------

        return dY
