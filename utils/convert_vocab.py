from typing import List


class ConvertVocab():
    @staticmethod
    def word2id(vocab: List[str]):
        """
        Converts a vocab list of string to a Dictonary
        @Return word2id, id2word (Dict[str, int], Dict[int, str])
        """
        word2id = {key: value for key, value in enumerate(vocab)}
        id2word = {value: key for key, value in word2id.items()}
        return word2id, id2word
