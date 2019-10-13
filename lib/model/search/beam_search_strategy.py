from math import log

from IPython.display import display

from lib.utils.array_utils import args_max


def end_with(seq, word_seq):
    return len(seq[0]) > 1 and seq[0][-1] == word_seq


def is_completed(seq, seq_end):
    return len(completed([seq], seq_end)) > 0


def completed(sequences, seq_end):
    return [seq for seq in sequences if end_with(seq, seq_end)]


def all_completed(sequences, seq_end):
    return len(completed(sequences, seq_end)) == len(sequences)


class BeamSearchStrategy:
    def __init__(
            self,
            model,
            sequencer,
            seq_prefix,
            seq_postfix,
            k,
            verbose=True
    ):
        self.model = model
        self.sequencer = sequencer
        self.seq_prefix = seq_prefix
        self.seq_postfix = seq_postfix
        self.k = k
        self.seq_end = self.sequencer.to_seq(self.seq_postfix)[0]
        self.__verbose = verbose

    def __predict(self, image_feature, sequence):
        return self.model.predict([image_feature, self.sequencer.pad(sequence)], verbose=0)

    def perform(self, image_feature):
        image_feature = image_feature.reshape((1, 2048))
        sequences = [([], 1.0)]

        while not all_completed(sequences, self.seq_end):
            candidates = []
            for sequence in sequences:
                seq, score = sequence
                if is_completed(sequence, self.seq_end):
                    candidates.append(sequence)
                    continue

                word_dist = self.__predict(image_feature, seq)[0]

                top_word_ind = args_max(word_dist, top=self.k)
                for word_ind in top_word_ind:
                    candidate = (seq + [word_ind], score * word_dist[word_ind])
                    candidates.append(candidate)

            sequences = sorted(candidates, key=lambda tup: tup[1], reverse=True)[:self.k]
            if self.__verbose:
                display(self.__as_phrases(sequences))

        return self.__as_phrases(sorted(sequences, key=lambda tup: tup[1], reverse=True))

    def __as_phrases(self, sequences):
        return [(self.sequencer.to_phrase(seq).replace(self.seq_postfix, '').strip(), score) for seq, score in
                sequences]
