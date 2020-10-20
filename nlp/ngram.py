#!/usr/bin/env python

"""Implements a simple Markov text generator"""

from __future__ import print_function
from collections import Counter
import string
from sys import argv
import textwrap
import numpy as np

class NGram(object):
    """NGram generator"""
    def __init__(self, text, n=4):
        self.n = n
        self.frequencies = self.get_frequencies(text)

    def get_frequencies(self, reference):
        """
        Returns a dict containing the number of n-grams found within the reference
        """
        last_ind = len(reference) - self.n + 1
        return Counter(reference[i: i+self.n] for i in range(last_ind))

    def weighted_pick(self, elements):
        """
        Picks an element from list elements where each index corresponds to a number
        in frequencies.
        """
        weights = np.array([self.frequencies.get(e, 0) for e in elements], dtype=np.float64)
        weights /= sum(weights) # Ensures everything sums to 1

        return np.random.choice(elements, p=weights)

    def get_next_letter(self, sentence):
        """
        Gets the next letter of the string based on an n-gram using dict frequencies.
        """
        last_letters = sentence[-self.n + 1:] # String containing n-1 letters
        new_elements = []
        for char in string.printable:
            key = last_letters + char
            new_elements.append(key)
        return self.weighted_pick(new_elements)[-1]

    def generate_seed(self):
        keys = self.frequencies.keys()
        new_elements = [key for key in keys if key[0] in string.ascii_uppercase]
        return self.weighted_pick(new_elements)

    def produce_sentence(self):
        """
        Produces a sentence using n-grams with given reference.
        """
        sentence = self.generate_seed()
        while "." not in sentence:
            sentence += self.get_next_letter(sentence)
        return sentence

def parse_arguments():
    try:
        path = argv[1]
    except IndexError:
        print("usage: python3 "+__name__+" <path> <n=4>")
        return
    try:
        n = int(argv[2])
    except IndexError:
        n = 4
    print("Loading...", end="")
    with open(path) as f:
        text = f.read()
    return text, n


def main():
    text, n = parse_arguments()
    wrapper = textwrap.TextWrapper()
    ngram = NGram(text, n=n)
    print("Done. Press enter to generate text or type 'exit' to leave")
    while input() != 'exit':
        print(wrapper.fill(ngram.produce_sentence()))

if __name__ == "__main__":
    main()
