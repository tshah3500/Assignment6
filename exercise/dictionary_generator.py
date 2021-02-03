#Running this file produces the file spanish_to_english.dat
#If you want to add some new words to the dictionary, feel
#free to do so, then save, and rerun this file. 

import pickle
from typing import Tuple, List, Dict
import sys

spa_dict = {
    "gato": "cat",
    "perro": "dog",
    "universidad" : "college",
    "amor" : "love",
    "computadora": "computer",
    "ciencias": "science",
    "comida":"food",
    "agua": "water",
    "calculadora": "calculator"
}

def save(dict: Dict, filename: str) -> None:
    """Pickles given dictionary to a file with the given name

    Args:
      dict - a dictionary to pickle
      filename - relative path to file to save
    """
    with open(filename, 'wb') as f:
      pickle.Pickler(f).dump(dict)

save(spa_dict,"spanish_to_english.dat")
