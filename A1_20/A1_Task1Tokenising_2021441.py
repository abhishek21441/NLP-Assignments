from A1_Task1Encoding_2021056 import get_stats, merge_vocab,process_string,read_text_file

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