#!/usr/bin/env python3

##########
# Just writing a bit of Python code to make sure I have
# my satisfaction on this new macbook. Github working, etc.
##########
import sys
import logging
import argparse

FORMAT = '%(levelname)|%(asctime)-15s| %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)


def main():

    logging.info("BEGIN: coke.py")

    parser = argparse.ArgumentParser()
    parser.add_argument("sing", action='store_true')
    args = parser.parse_args()

    song = """The sun will always shine.
The birds will always sing.
As long as there is thirst,
There's always the real thing.
"""

    if args.sing:
        logging.info("Singing")
        print(song)
    else:
        logging.info("Not singing")

    logging.info("END: coke.py")

if __name__ == '__main__':
    main()
