_DEFAULT = 0
_PAST_VERB = 1  

contractions = { 
    "ain't": "is not", 
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",  # based on next VERB => he had / he would ::: (PAST / DEFAULT)
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",  # based on next WORD => he is / he has ::: (DEFAULT / PAST)
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",  # Q based on next WORD ==> how has / how is / how does (Solved by assuming most probable)
    "i'd": "i would",  # based on next VERB => i had / i would ::: (PAST / DEFAULT)
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",  # based on next VERB ==> it had / it would ::: (PAST / DEFAULT)
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",  # based on next WORD => it is / it has ::: (DEFAULT / PAST)
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",  # based on next VERB ==> she had / she would ::: (PAST / DEFAULT)
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",  # based on next WORD => she is / she has ::: (DEFAULT / PAST)
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",  # based on next VERB ==> that would / that had ::: (PAST / DEFAULT)
    "that'd've": "that would have",
    "that's": "that is",  # based on next WORD => that is / that has ::: (DEFAULT / PAST)
    "there'd": "there would",  # based on next VERB ==> there had / there would ::: (PAST / DEFAULT)
    "there'd've": "there would have",
    "there's": "there is",  # based on next WORD => there is / there has ::: (DEFAULT / PAST)
    "they'd": "they would",  # based on next VERB ==> they had / they would ::: (PAST / DEFAULT)
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",  # based on next VERB ==> we had / we would ::: (PAST / DEFAULT)
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",  # Q based on next WORD ==> what has / what is (Solved by assuming most probable)
    "what've": "what have",
    "when's": "when is",  # Q based on next WORD ==> when has / when is (Solved by assuming most probable)
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",  # Q based on next WORD ==> where has / where is (Solved by assuming most probable)
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",  
    "who's": "who is",  # Q based on next WORD ==> who has / who is (Solved by assuming most probable)
    "who've": "who have",
    "why's": "why is",  # Q based on next WORD ==> why has / why is (Solved by assuming most probable)
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",  # based on next VERB ==> you had / you would ::: (PAST / DEFAULT)
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
}
      
past_contractions = {
    "he'd": "he had",
    "i'd": "i had",
    "it'd": "it had",
    "she'd": "she had",
    "that'd": "that had",
    "they'd": "they had",
    "we'd": "we had",
    "you'd": "you had",
    "there'd": "there had",

    "he's": "he has",
    "it's": "it has",
    "she's": "she has",
    "that's": "that has",
    "there's": "there has",
}


def expand(tagged_words):
    """ Acceptes list of tagged words to expand abbreviations within it """
    words = []
    for i in range(len(tagged_words)):
        next_word_type = None
        if "'" in tagged_words[i][0]:
            if tagged_words[i][0] in contractions:
                if i < len(tagged_words) - 1:
                    next_word_type = _tag_to_word_type(tagged_words[i + 1][1])
                words += _expand_word(tagged_words[i][0], next_word_type)
            elif i != 0:
                w = tagged_words[i - 1][0] + tagged_words[i][0]
                if w in contractions:
                    if i < len(tagged_words) - 1:
                        next_word_type = _tag_to_word_type(tagged_words[i + 1][1])
                    del words[-1]
                    words += _expand_word(w, next_word_type)
                else:
                    words.append(tagged_words[i][0])
            else:
                words.append(tagged_words[i][0])
        else:
            words.append(tagged_words[i][0])

    return words

def _tag_to_word_type(tag=None):
    """ Based on http://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html """

    if tag is None:
        return _DEFAULT
    elif tag == "VBD" or tag == "VBN":
        return _PAST_VERB
    return _DEFAULT

def _expand_word(word, next_word_type):
    """ Returns List of word(s) representing the expansion of a single word """

    if next_word_type == _PAST_VERB and word in past_contractions:
        return past_contractions[word].split()
    elif word in contractions:
        return contractions[word].split()

    return [word]

