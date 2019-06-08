from lib.model.search.beam_search_strategy import BeamSearchStrategy
from lib.model.sequencer import Sequencer


class SearchFactory:

    @staticmethod
    def beam_search(
            model,
            word_to_index,
            index_to_word,
            max_seq_len,
            seq_prefix='$',
            seq_postfix='#',
            k=7,
            verbose=True
    ):
        return BeamSearchStrategy(
            model=model,
            sequencer=Sequencer(word_to_index, index_to_word, max_seq_len),
            seq_prefix=seq_prefix,
            seq_postfix=seq_postfix,
            k=k,
            verbose=verbose
        )
