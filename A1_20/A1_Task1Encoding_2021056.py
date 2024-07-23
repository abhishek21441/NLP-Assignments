from A1_Task1Tokenising_2021441 import learn_vocabulary,tokenize


"""Code to convert corpus to format readable by BPE Algorithm"""
def read_text_file(file_path):
    word_freq = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        #Checks casing and punctuation if any
        content = ''.join(char.lower() if char.isalpha() or char.isspace() else ' ' for char in content) 
        words = content.split()
    for word in words:
                word2 = ''
                for i in word:
                       word2 += i + ' ' # Separates each letter of the word by space and ends with $ as separating token
                word2 += '$'
                word_freq[word2] = word_freq.get(word2, 0) + 1 # Updates word frequency
    return word_freq

import string
def process_string(input_string):
    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    no_punctuation = input_string.translate(translator)

    # Convert to lowercase
    lowercase_string = no_punctuation.lower()

    return lowercase_string


"""Code for Byte Pair Algorithm"""
# Byte Pair Encoding as per research Paper
import re, collections
def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for idx, (word, freq) in enumerate(vocab.items()):
        symbols = word.split()
        symbol_length = len(symbols)
        for i in range(0,symbol_length-1,1):
            pairs[symbols[i],symbols[i+1]] = pairs[symbols[i],symbols[i+1]] + freq # Check & update frequency of all pairs formed in corpus
    return pairs
    
def merge_vocab(pair, v_in):

    bigram = re.escape(' '.join(pair))
    v_out = {}
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for idx, word in enumerate(v_in):
        w_out = p.sub(''.join(pair), word) # Add best pair to the existing vocabulary
        v_out[w_out] = v_in[word]
    return v_out

# if __name__ == "__main__":
#     file_path = 'corpus.txt'
#     # s = input("Enter corpus name: ")
#     # file_path = s
#     vocab = read_text_file(file_path)
#     print("Conveted corpus to dictionary format as readable by BPE.")


#     merge_rules = [] # Stores the different merges in order of best(first) to last
#     all_tokens = set() # Stores cumulatively all the tokens formed during encoding the corpus

#     num_merges = 30000
#     # s = int(input("Enter number of merges: "))
#     # num_merges = s

#     # Add all the unique letters in the vocab initially
#     initial_words = list(vocab.keys())
#     for i in range(len(initial_words)):
#         for j in initial_words[i]:
#             if j != ' ':
#                 all_tokens.add(j)
    
#     merge_rules = learn_vocabulary(num_merges=num_merges,all_tokens=all_tokens,merge_rules=merge_rules,vocab=vocab)

#     """
#     3. Taking in input sentences from a given samples.txt file, tokenising them based on above merge rules and finally storing the output in tokenised_samples.txt
#     """
#     with open('Samples.txt', 'r', encoding='utf-8') as input_file:
#         lines = [line.strip() for line in input_file.readlines()]

#     with open('tokenized_samples.txt', 'w', encoding='utf-8') as output_file:
#         for line in lines:
#             process_line = process_string(line)
#             ansline = tokenize(process_line,merge_rules)
#             output_file.write(f"{ansline}\n")

#     print("Lines processed and written to tokenized_samples.txt")