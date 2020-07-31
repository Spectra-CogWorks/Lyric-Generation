import numpy as np
from collections import Counter, defaultdict
import re
import pickle

# loads database of language models, organized by artist
with open("artist_to_lm.pkl", mode="rb") as opened_file:
    artist_to_lm = pickle.load(opened_file)

# loads database of lyrics, organized by artist
with open("name_to_lyr.pkl", mode="rb") as opened_file:
    name_to_lyr = pickle.load(opened_file)


def unzip(pairs):
    """
    "Unzips" groups of items into separate tuples.
    
    Parameters
    ----------
    pairs : Iterable[Tuple[Any, ...]]
        An iterable of the form ((a0, b0, c0, ...), (a1, b1, c1, ...))
    
    Returns
    -------
    Tuple[Tuples[Any, ...], ...]
       A tuple containing the "unzipped" contents of "pairs";
       i.e. ((a0, a1, ...), (b0, b1, ...), (c0, c1), ...)
    """
    return tuple(zip(*pairs))


def normalize(counter):
    """
    Convert a "word -> count" counter to a list of (word, frequency) pairs,
    sorted in descending order of frequency.

    Parameters
    ----------
    counter : collections.Counter
        A counter with the format "word -> count".

    Returns
    -------
    List[Tuple[str, int]]
       A list of tuples (word, frequency) in order of descending frequency.
    """
    total = sum(counter.values())
    return [(char, cnt/total) for char, cnt in counter.most_common()]


def train_lm(lyrics, n):
    """ 
    Train a word-based n-gram language model.

    Parameters
    ----------
    lyrics: str or List[str]
        A string or list of strings (doesn't need to be lowercase) representing the lyrics you want
        to train the model on.

    n: int
        The length of n-gram to analyze.

    Returns
    -------
    Dict[str, List[Tuple[str, float]]] : {n-1 history -> [(letter, normalized count), ...]}
        A dict that maps histories (strings of length (n-1)) to lists of (word, prob) pairs,
        where "prob" is the probability/frequency of "word" appearing after that specific history.
    """
    # initializes defaultdict with Counter type
    model = defaultdict(Counter)
    
    # splits up text into "tokens" of individual words
    if type(lyrics) == list:
        tokens = []
        for item in lyrics:
            tokens.append(tuple(re.split(' |\n|\t', item)))            
    else:
        tokens = re.split(' |\n|\t', lyrics)
    
    # iterates through tokens to populate lm
    for item in tokens:
        history = "~ " * (n-1)
        # if there were multiple songs in input lyrics
        if type(item) == tuple:
            for word in item:
                # skips over tokens like "[Chorus]" or ""
                if "[" in word or "]" in word or "" == word:
                    continue
                else:
                    # given a history, adds to the count of the next word
                    model[history][word] += 1
                    # slides history over by one word
                    if history[-1] == " ":
                        history = " ".join(history.split(" ")[1:]) + word
                    else:
                        # adds a space between the history and the new word if needed
                        history = " ".join(history.split(" ")[1:]) + " " + word
        # if there was only one song in input lyrics
        else:
            # skips over tokens like "[Chorus]" or ""
            if "[" in item or "]" in item or "" == item:
                continue
            else:
                # given a history, adds to the count of the next word
                model[history][item] += 1
                # slides history over by one word
                if history[-1] == " ":
                    history = " ".join(history.split(" ")[1:]) + item
                else:
                    # adds a space between the history and the new word if needed
                    history = " ".join(history.split(" ")[1:]) + " " + item
    
    # normalizes the word counts by dividing by the total count of words that appeared after each history
    lm = {history : normalize(counter) for history, counter in model.items()}
    
    return lm


def generate_word(lm, history):
    """
    Randomly picks word according to probability distribution associated with 
    the specified history, as stored in the language model.

    Parameters
    ----------
    lm: Dict[str, List[Tuple[str, float]]] 
        The n-gram language model, with format: history -> [(word, freq), ...]

    history: str
        A string of length (n-1) to use as context/history for generating the next word.

    Returns
    -------
    str
        The predicted next word, or '~' if history is not in language model.
    """
    if history not in lm:
        return '~'
    
    word, prob = unzip(lm[history])
    return np.random.choice(word, p=prob)


def generate_text(original, lm, n, n_words=100):
    """ 
    Randomly generates n_words of text by drawing from the probability distributions stored in n-gram language model lm.

    Parameters
    ----------
    original: str or List[str]
        The original song lyrics used to train the model.
    
    lm: Dict[str, List[Tuple[str, float]]]
        The trained n-gram language model, with format: history -> [(char, freq), ...]
        
    n: int
        Order of n-gram model.

    n_words: int
        Number of words to randomly generate.

    Returns
    -------
    str
        Text generated by the trained model.
    """
    # initialize history as sequence of ~'s (with spaces so they can be tokenized)
    history = "~ " * (n - 1)
    
    text = []
    
    # repeats generate_word() function for the number of words the user wants
    for i in range(n_words):
        word = generate_word(lm, history)
        text.append(word)
        # creates a new history by dropping the first word and adding the newly generated one
        if history[-1] == " ":
            history = " ".join(history.split(" ")[1:]) + word
        else:
            # adds a space between the history and the new word if needed
            history = " ".join(history.split(" ")[1:]) + " " + word
    
    # removes parentheses from generated words
    for i in range(len(text)):
        if "(" in text[i]:
            text[i] = text[i][1:]
        
    # removes parentheses from generated words
    for i in range(len(text)):
        if ")" in text[i]:
            text[i] = text[i][:-1]
    
    # if the word was the beginning of a new line in the original lyrics, makes it the beginning of the new line in the generated lyrics as well
    for i in range(1, len(text)):
        # if there were multiple songs in input lyrics
        if type(original) == list:
            for item in original:
                if "\n" + text[i] in item and text[i][0] != "\n":
                    text[i] = "\n" + text[i]
        # if there was only one song in input lyrics
        else:
            if "\n" + text[i] in original:
                text[i] = "\n" + text[i]
    
    # joins words in list "text" with spaces to display generated lyrics
    return print(" ".join(text))


def start():
    """
    Runs the command line interface.

    Parameters
    ----------
    None

    Returns
    -------
    None    
    """
    # asks to input an artist from the list
    artist = input(
                """\nPlease enter an artist. Your options are: Adele, Alicia Keys, Beatles, Bieber, Bob Dylan, Britney Spears, Bruce Springsteen, Bruno Mars,
Disney, Dolly Parton, Drake, Eminem, Janis Joplin, Johnny Cash, Lady Gaga, Lin-Manuel Miranda, Lorde, Michael Jackson, Prince, and Rihanna:\n""")
    
    # reformats the input so it can be called from the databases
    artist = artist.lower()
    if " " in artist:
        artist = "-".join(artist.split(" "))

    # if user enters an artist not in the database
    if artist not in artist_to_lm.keys():
        ans = input("Sorry, that artist is not in my database. Would you like to try again? [Y/N] ")
        # restarts program if user wants to
        if ans in ["Y", "y"]:
            start()
    else:
        # asks user how many lyrics to generate, then generates
        n_words = int(input("Please enter the number of words you want to generate: "))
        generate_text(name_to_lyr[artist], artist_to_lm[artist], 3, n_words=n_words)
        
        # restarts program if user wants to
        again = input("\nLyric generation complete! Do you want to try again? [Y/N] ")
        if again in ["Y", "y"]:
            start()

# runs by default when this python file is opened in the command line
start()