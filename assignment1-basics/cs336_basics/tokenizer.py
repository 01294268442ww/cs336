import pickle
import regex as re
from collections import defaultdict

def train_bpe(input_path, vocab_size=10000, special_tokens=["<|endoftext|>"]):
    """
    Write a function that, given a path to an input text file, trains a (byte-level) BPE tokenizer.

    input_path: str Path to a text file with BPE tokenizer training data.

    vocab_size: int A positive integer that defines the maximum final vocabulary size (including the
    initial byte vocabulary, vocabulary items produced from merging, and any special tokens).

    special_tokens: list[str] A list of strings to add to the vocabulary. These special tokens do not
    otherwise affect BPE training.

    return:
    vocab: dict[int, bytes] The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes).

    merges: list[tuple[bytes, bytes]] A list of BPE merges produced from training. Each list item
    is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with
    <token2>. The merges should be ordered by order of creation.
    """

    print("entry train_bpe")

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    vocab = {i: bytes([i]) for i in range(256)}
    next_id = 256

    special_token_bytes = [token.encode("utf-8") for token in special_tokens]
    for token_bytes in special_token_bytes:
        if token_bytes not in vocab.values():
            vocab[next_id] = token_bytes
            next_id += 1

    # Step 2: Pre-tokenization
    print("Step 2: Pre-tokenization")
    pre_tokens_cnt = defaultdict(int)

    def to_bytes_tuple(word: str):
        l = list(tuple(word.encode("utf-8")))
        l = [bytes([x]) for x in l]
        return tuple(l)

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    chunks = re.split("|".join(map(re.escape, special_tokens)), text)
    
    for chunk in chunks:
        for m in re.finditer(PAT, chunk):
            word = m.group(0)
            pre_tokens_cnt[to_bytes_tuple(word)] += 1   # key of pre_tokens_cnt e.g. (b'H', b'e', b'l', b'l', b'o')

    # Step 3: Compute BPE Merges
    print("Step 3: Compute BPE Merges")
    merges = []

    while len(vocab) < vocab_size:
        pair_counts = defaultdict(int)

        # Count all adjacent byte pairs
        for token, cnt in pre_tokens_cnt.items():
            for i in range(len(token) - 1):
                pair = (token[i], token[i + 1])
                pair_counts[pair] += cnt

        if not pair_counts:
            break  # No more pairs to merge

        # Find the most frequent pair(s)
        max_count = max(pair_counts.values())
        candidates = [k for k, v in pair_counts.items() if v == max_count]
        best_pair = max(candidates)

        a, b = best_pair

        # Create new token
        new_token = a + b
        vocab[next_id] = new_token
        next_id += 1

        # Apply the merge to all pre-tokenized sequences
        changes = []
        for token, cnt in pre_tokens_cnt.items():
            # Find all occurrences of the `best_pair` in `token`
            indices = [i for i in range(len(token) - 1) if token[i:i + 2] == best_pair]
            if indices:
                # Replace each occurrence with `new_token`
                new_pre_token = []
                i = 0
                while i < len(token):
                    if i in indices:
                        new_pre_token.append(new_token)
                        i += 2
                    else:
                        new_pre_token.append(token[i])
                        i += 1
                new_pre_token = tuple(new_pre_token)
                changes.append((token, new_pre_token, cnt))

        # apply change
        for old_token, new_pre_token, cnt in changes:
            pre_tokens_cnt[new_pre_token] = pre_tokens_cnt.get(new_pre_token, 0) + cnt
            del pre_tokens_cnt[old_token]

        # Record the merge
        merges.append((a, b))

    print("return")
    return vocab, merges


class Tokenizer:
    """
    Implement a Tokenizer class that, given a vocabulary and a list of merges, 
    encodes text into integer IDs and decodes integer IDs into text. 
    Your tokenizer should also support user-provided special tokens
    (appending them to the vocabulary if they aren’t already there)
    """
    def __init__(self, vocab, merges, special_tokens=None):
        """
        vocab: dict[int, bytes]
        merges: list[tuple[bytes, bytes]]
        special_tokens: list[str] | None = None
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.byte_2_id_vocab = {v:k for k, v in vocab.items()}

        self.merges_ranking = {merge:i for i, merge in enumerate(merges)}
        self.EOS_TOKEN = self.vocab.get(b"<|endoftext|>", None)

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """
        vocab_filepath: str
        merges_filepath: str
        special_tokens: list[str] | None = None
        """
        with open(vocab_filepath, "rb") as vf:
            raw_vocab = pickle.load(vf)

        norm_vocab = {}
        for k, v in raw_vocab.items():
            k = int(k)
            if isinstance(v, str):
                v = v.encode("utf-8")
            norm_vocab[k] = v

        with open(merges_filepath, "rb") as mf:
            raw_merge = pickle.load(mf)

        norm_merge = []
        for a, b in raw_merge:
            if isinstance(a, str):
                a = a.encode("utf-8")
            if isinstance(b, str):
                b = b.encode("utf-8")
            norm_merge.append((a, b))
        
        return cls(norm_vocab, norm_merge, special_tokens)
    
    def _pre_tokenize(self, text, special_tokens):
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        # frist split string
        if not special_tokens:
            return re.findall(PAT, text)
        
        tokens = sorted(special_tokens, key=len, reverse=True)
        union = "|".join(re.escape(t) for t in tokens)
        chunks = re.split(f"({union})", text)

        out = []
        st = set(special_tokens)
        for chunk in chunks:
            if not chunk:
                continue
            if chunk in st:
                out.append(chunk)
            else:
                out.extend(re.findall(PAT, chunk))
        
        return out
    
    def _word_2_bytes(self, word):
        word = list(word.encode("utf-8"))
        word_bytes = [bytes([b]) for b in word]

        return tuple(word_bytes)
    
    def _get_pair(self, word):
        """
        We can obtain all the adjacent combinations of a word.
        """
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        
        return pairs
    
    def _apply_the_merges(self, word_bytes):
        """
        We then take the sequence of vocabulary element merges created during BPE training, 
        and apply it to our pre-tokens in the same order of creation
        """

        # word_bytes eg 9 7 1 2
        # we need follow merges order to merge until don't merge to all words
        # word
        word = list(word_bytes)
        word_pairs = self._get_pair(word)
        if not word_pairs:
            return word
        
        while True:
            next_merge = min(word_pairs, key=lambda pair : self.merges_ranking.get(pair, float('inf')))

            if next_merge not in self.merges_ranking:
                break

            first, second = next_merge
            new_word = []
            i = 0

            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
            if len(word) == 1:
                break

            word_pairs = self._get_pair(word)
        
        return word

    
    def encode_text(self, pre_token):
        word_byte = self._word_2_bytes(pre_token)
        word_byte_after_merge = self._apply_the_merges(word_byte)
        token_ids = []
        for merged_bytes in word_byte_after_merge:
            id = self.byte_2_id_vocab[merged_bytes]
            token_ids.append(id)
        
        return token_ids

    def encode(self, text):
        """Encode an input text into a sequence of token IDs."""
        
        token_ids = []
        pretokenization = self._pre_tokenize(text, self.special_tokens)
        for part in pretokenization:
            if self.special_tokens and part in self.special_tokens:
                special_id = self.byte_2_id_vocab[part.encode("utf-8")]
                token_ids.append(special_id)
            else:
                token_ids.extend(self.encode_text(part))
        
        return token_ids


    def encode_iterable(self, iterable):
        """
        Given an iterable of strings (e.g., a Python file handle), 
        return a generator that lazily yields token IDs. 
        This is required for memory-efficient tokenization of large files 
        that we cannot directly load into memory
        """
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids):
        """Decode a sequence of token IDs into text"""
        byte_list = b''.join(self.vocab[id] for id in ids)

        return byte_list.decode("utf-8", errors="replace")



def main():
    vocab, merges = train_bpe("data/TinyStoriesV2-GPT4-valid.txt")
    with open("data/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    
    with open("data/merges.pkl", "wb") as f:
        pickle.dump(merges, f)


if __name__ == "__main__":
    print("start main")
    main()
    print("end main")
