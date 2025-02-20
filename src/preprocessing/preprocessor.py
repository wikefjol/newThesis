from typing import Protocol, List, Any
#from utils.logging_utils import with_logging
from src.utils.vocab import Vocabulary
class Strategy(Protocol):
    '''Augments sequence by imitating sequencing errors'''
    def execute(self, sequence: list[str]) -> list[str]:
        """
        Parameters
        ----------
        sequence : str
            DNA sequence

        Returns
        ----------
        sequence : str
            Augmented DNA sequence
        """

class Preprocessor:
    def __init__(
        self,
        augmentation_strategy: Strategy,
        tokenization_strategy: Strategy,
        padding_strategy: Strategy,
        truncation_strategy: Strategy,
        vocab: Vocabulary = None
    ):
        self.augmentation_strategy = augmentation_strategy
        self.tokenization_strategy = tokenization_strategy
        self.padding_strategy = padding_strategy
        self.truncation_strategy = truncation_strategy
        self.vocab = vocab

    #@with_logging(level=10)
    def process(self, sequence: str) -> List[List[str]]:
        sequence = list(sequence)  # Convert string to list of characters
        
        # Step 1: Augmentation
        augmented_sequence: List[str] = self.augmentation_strategy.execute(sequence)
        #print(f"\n[Step 1: Augmented Sequence]\n{augmented_sequence}")

        # Step 2: Tokenization
        tokenized_sentence: List[List[str]] = self.tokenization_strategy.execute(augmented_sequence)
        #print(f"\n[Step 2: Tokenized Sentence]\n{tokenized_sentence}")

        # Step 3: Padding
        padded_sentence: List[List[str]] = self.padding_strategy.execute(tokenized_sentence)
        #print(f"\n[Step 3: Padded Sentence]\n{padded_sentence}")

        # Step 4: Truncation
        processed_sentence: List[List[str]] = self.truncation_strategy.execute(padded_sentence)
        #print(f"\n[Step 4: Processed (Post-Truncation) Sentence]\n{processed_sentence}")

        mapped_sentence: List[List[int]] = self.vocab.map_sentence(processed_sentence)
        return mapped_sentence

class OverlappingPreprocessor:
    def __init__(
        self,
        k: int,
        augmentation_strategy: Strategy,
        tokenization_strategy: Strategy,
        padding_strategy: Strategy,
        truncation_strategy: Strategy,
        vocab: Vocabulary = None
    ):
        self.k = k
        self.augmentation_strategy = augmentation_strategy
        self.tokenization_strategy = tokenization_strategy
        self.padding_strategy = padding_strategy
        self.truncation_strategy = truncation_strategy
        self.vocab = vocab

    def process(self, sequence: str) -> List[List[int]]:
        sequence_list = list(sequence)

        # Step 1: Augmentation
        augmented_sequence: List[str] = self.augmentation_strategy.execute(sequence_list)

        # Step 1.5: Create overlapping windows using self.k
        overlapping_sequences: List[List[str]] = [augmented_sequence[i:] for i in range(self.k)]
        Debug: print(f"[Overlapping Sequences]: {overlapping_sequences}")

        # Step 2: Tokenization for each overlapping window.
        tokenized_windows = [self.tokenization_strategy.execute(seq) for seq in overlapping_sequences]
        Debug: print(f"[Tokenized Windows]: {tokenized_windows}")

        # Step 2.5: Flatten the tokenized windows.
        #SEP = [self.vocab.get_token('SEP')]
        #print(f"SEP is: {SEP} and type is: {type(SEP)}")
        combined_tokens = []
        for window in tokenized_windows:
            for token in window:
                combined_tokens.append(token)
            combined_tokens.append(['SEP'])
        combined_tokens = combined_tokens[:-1] # Removes the last "SEP"
        #print(f"First token is: {combined_tokens[0]} and of type: {type(combined_tokens[0])}")

        # Step 3: Padding
        padded_sentence: List[str] = self.padding_strategy.execute(combined_tokens)
        Debug: print(f"[Padded Sentence]: {padded_sentence}")

        # Step 4: Truncation
        processed_sentence: List[str] = self.truncation_strategy.execute(padded_sentence)
        Debug: print(f"[Processed Sentence]: {processed_sentence}")

        mapped_sentence: List[List[int]] = self.vocab.map_sentence(processed_sentence)
        return mapped_sentence
        





    
