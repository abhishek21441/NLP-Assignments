"""
This Jupyter Notebook runs all the parts for Task 1. This includes the following:-
1. Generating the tokens from the given corpus and storing them in tokens.txt
2. Learning the merge rules using the Byte Pair Algorithm and storing the rules in merge_rules.txt
3. Taking in input sentences from a given samples.txt file, tokenising them based on above merge rules and finally storing the output in tokenised_samples.txt
"""

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


"""Code for tokenising example sentences based on merge rules learnt"""
def merge_once(temp_state, merge_rules): # Applies one iteration of merge rule, i.e. atmost one rule applied
    if len(temp_state) == 1:
        return temp_state
    temp2 = []
    flag = 0
    for i in range(len(temp_state)-1): # We want to check pairs like 'l','o' or 'o','w' and so on
        if flag == 0:
            for j in merge_rules: # Iterate through each rule
                if temp_state[i] == j[0] and temp_state[i+1] == j[1]:
                    temp2.append("".join(map(str, j))) # Conduct one join
                    flag = 1 # set flag
                    break
            if flag == 0: # if no join found for 1st letter
                temp2.append(temp_state[i]) # add that letter to temp2 as is
                # imp incase 2nd last letter is added without flag=1 then add last letter aswell
                if i == len(temp_state)-2:
                    temp2.append(temp_state[-1])
        else: # Once flag is set, simply add the rest of the list as only one join allowed at a time
            temp2 += temp_state[i+1:]
            break
    return temp2
        
def merge_word(word,merge_rules): # Given a single word returns a list of its compositional tokens
    # Given a word like 'low' returns ['low']
    # Given a word like 'widest' returns ['wide','s','t']
    temp_state = [i for i in word] # breaks into list of chars ===> ['l', 'o', 'w']
    old = temp_state
    len2 = 0
    len1 = len(old)
    while len2 != len1: # if the merge algo makes no change then we have tokenised successfully
        len1 = len(old)
        new = merge_once(old,merge_rules) # Does exactly one merge
        len2 = len(new)
        old = new.copy() # Now set the old to current output and new to the output we'll get passing the currrent to merge
    return new

def merge_sentence(sample_sentence,merge_rules): # Given a sentence returns its tokens separated by comma
    # Splits the words by whitespace e.g. low widest z => ['low','widest','z'] 
    sample_sentence = sample_sentence.split() 
    # Given a sample sentence like ['low','widest','z'] => ['low','wide','s','t','z']
    sample_sentence2 = []
    for i in sample_sentence:
        sample_sentence2.append(i+"$")
    final_sent = []
    for word in sample_sentence2: # a single word ===> 'widest'
        new = merge_word(word,merge_rules) # returns ==> ['wide','s','t']
        final_sent += new
    return final_sent

def tokenize(sample_sentence,merge_rules): # simply concats the list provided by merge sentence
    line = ','.join(merge_sentence(sample_sentence,merge_rules))
    return line


"""Learning the merge rules and building the vocabulary"""
def learn_vocabulary(num_merges,all_tokens,vocab,merge_rules):
    for i in range(num_merges):
        pairs = get_stats(vocab)
        if len(pairs) == 0:
            print(i)
            break # Incase no new pairs form means we have completed as high a level of token as possible

        best = max(pairs, key=pairs.get) # pair with the most number of instances

        merge_rules.append(best) # Add the current best pair as a rule

        vocab = merge_vocab(best, vocab) 

        new_token = "".join(map(str, best)) # Make the best pair as a new token formed at that stage
        all_tokens.add(new_token) # Add the new token to the list of all tokens
    all_tokens = list(all_tokens)
    def custom_sort_key(s):
        return (len(s), s)
    all_tokens = sorted(all_tokens, key=custom_sort_key) # Writing all tokens in order of length and ASCII value


    # Make text files

    # Making tokens.txt
    tokens_file_path = 'tokens.txt'
    with open(tokens_file_path, 'w', encoding='utf-8') as file:
        for token in all_tokens:
            file.write(f"{token}\n")

    print(f"Tokens written to {tokens_file_path}")

    # Making merge_rules.txt
    rules_file_path = 'merge_rules.txt'
    with open(rules_file_path, 'w', encoding='utf-8') as file:
        for rules in merge_rules:
            rule = rules[0] + "," + rules[1]
            file.write(f"{rule}\n")


    print(f"Rules written to {rules_file_path}")

    return merge_rules

"""
1. Generating the tokens from the given corpus and storing them in tokens.txt
2. Learning the merge rules using the Byte Pair Algorithm and storing the rules in merge_rules.txt
"""

if __name__ == "__main__":
    file_path = 'corpus.txt'
    # s = input("Enter corpus name: ")
    # file_path = s
    vocab = read_text_file(file_path)
    print("Conveted corpus to dictionary format as readable by BPE.")


    merge_rules = [] # Stores the different merges in order of best(first) to last
    all_tokens = set() # Stores cumulatively all the tokens formed during encoding the corpus

    num_merges = 300
    # s = int(input("Enter number of merges: "))
    # num_merges = s

    # Add all the unique letters in the vocab initially
    initial_words = list(vocab.keys())
    for i in range(len(initial_words)):
        for j in initial_words[i]:
            if j != ' ':
                all_tokens.add(j)
    
    merge_rules = learn_vocabulary(num_merges=num_merges,all_tokens=all_tokens,merge_rules=merge_rules,vocab=vocab)

    """
    3. Taking in input sentences from a given samples.txt file, tokenising them based on above merge rules and finally storing the output in tokenised_samples.txt
    """
    with open('Samples.txt', 'r', encoding='utf-8') as input_file:
        lines = [line.strip() for line in input_file.readlines()]

    with open('tokenized_samples.txt', 'w', encoding='utf-8') as output_file:
        for line in lines:
            process_line = process_string(line)
            ansline = tokenize(process_line,merge_rules)
            output_file.write(f"{ansline}\n")

    print("Lines processed and written to tokenized_samples.txt")