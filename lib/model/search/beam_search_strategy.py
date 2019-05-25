from math import log
from lib.utils.array_utils import args_max, intersection
import numpy as np


def has_word(sequence, word_seq):
    return len(intersection(word_seq, sequence)) > 0


class CriteriaContext:
    def __init__(self, sequences, sequencer):
        self.sequences = sequences
        self.sequencer = sequencer


class AtLeastNEndWithWord:
    def __init__(self, n, word):
        self.n = n
        self.word = word

    def eval(self, ctx):
        word_seq = ctx.sequencer.to_seq(self.word)
        return len([1 for seq, _ in ctx.sequences if has_word(seq, word_seq)]) >= self.n


class MinWords:
    def __init__(self, count): self.count = count + 1

    def eval(self, ctx): return np.all([len(seq) >= self.count for seq, _ in ctx.sequences])


class MaxWords:
    def __init__(self, count): self.count = count + 1

    def eval(self, ctx): return np.all([len(seq) <= self.count for seq, _ in ctx.sequences])


class EndCriteria:
    def __init__(self): self.criteria = []

    def at_lest_n_end_with(self, n, word):
        self.criteria.append(AtLeastNEndWithWord(n, word))
        return self

    def min_words(self, count):
        self.criteria.append(MinWords(count))
        return self

    def max_words(self, count):
        self.criteria.append(MaxWords(count))
        return self

    def eval(self, ctx): return np.all([criterion.eval(ctx) for criterion in self.criteria])


class BeamSearchStrategy:
    def __init__(self,
                 model,
                 sequencer,
                 seq_prefix,
                 seq_postfix,
                 end_criteria,
                 k=3
                 ):
        self.model = model
        self.sequencer = sequencer
        self.seq_prefix = seq_prefix
        self.seq_postfix = seq_postfix
        self.k = k
        self.end_criteria = end_criteria

    def __predict(self, image_feature, sequence):
        return self.model.predict([image_feature, self.sequencer.pad(sequence)], verbose=0)

    def perform(self, image_feature):
        image_feature = image_feature.reshape((1, 2048))
        sequences = [([], 1.0)]

        while not self.end_criteria.eval(CriteriaContext(sequences, self.sequencer)):
            candidates = []
            for seq, score in sequences:
                word_dist = self.__predict(image_feature, seq)
                word_dist = word_dist[0]

                top_best_word_ind = args_max(word_dist, top=self.k)

                for word_ind in top_best_word_ind:
                    candidate = (seq + [word_ind], score * -log(word_dist[word_ind]))
                    candidates.append(candidate)

            sequences = sorted(candidates, key=lambda tup: tup[1])[:self.k]

        end_seq = self.sequencer.to_seq(self.seq_postfix)
        complete_sequences = [seq for seq in sequences if has_word(seq[0], end_seq)]
        best_sequences = sorted(complete_sequences, key=lambda tup: tup[1])
        return [(self.sequencer.to_phrase(seq).replace(self.seq_postfix, ''),  score) for seq, score in best_sequences]
