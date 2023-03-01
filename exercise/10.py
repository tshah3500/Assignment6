import pickle
from typing import Tuple, List, Dict
import sys

def load(filename: str) -> Dict:
    """Loads pickled dictionary stored in given file

    Args:
      filename - relative path to file to load

    Returns:
      dictionary stored in given file
    """
    with open(filename, "rb") as f:
      return pickle.Unpickler(f).load()

# Group Challenge
# I have provided you with a file called spanish_to_english.dat
# that represents a pickled dictionary where the keys are in spanish, 
# and the values are in english. (It only has around 10 words, 
# not by any means a complete dictionary unfortunately).
# Translation courtesy of Google Translate :) 

# Task 1 unpickle the dictionary and assigned it to the variable
# spa_eng_dict. 
# This should be 1 line of code that starts with...
# spa_eng_dict = ???

spa_eng_dict = load("spanish_to_english.dat")


assert spa_eng_dict["gato"] == "cat"
assert spa_eng_dict["perro"] == "dog"

words = ["gato", "universidad", "pitón", "amor", "ciencias", "comida"]
with open('translation.txt', 'w') as f:
  for eng in words:
      if eng in spa_eng_dict.keys():
        f.write(spa_eng_dict[eng])
      else:
        f.write('unkwown')
      f.write('\n')
  f.close()

# Task 2:
# Create a file named translation named translation.txt
# for each line, give the english translation of a spanish word
# that's in the words list. Make sure it's ordered accordingly.
# this means that the first line in the file should be cat, while the
# second line should be university. 
# if a word is not in the spa_eng_dict, then write "unknown" on the line.
# I have provided a file called translation_ans.txt to show you what
# the final result should look like. 


