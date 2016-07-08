#!/usr/bin/env python3

##########################################################
# Test code. We should use it to practice collaborating. #
# And to develop a baseline of coding standards.         #
##########################################################

import sys
import logging
import argparse

FORMAT = '%(levelname)s|%(asctime)-15s| %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)

def sing(out):
    """
    Writes a song to the specified output
    """
    song = """The sun will always shine.
The birds will always sing.
As long as there is thirst,
There's always the real thing.
"""
    out.write(song)



def main():

    logging.info("BEGIN: coke.py")
    description = """A toy program, but one to test things out on.
Can also perhaps serve as a baseline for coding standards."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--sing", action='store_true',
                        help="A flag to tell the code to sing.")
    parser.add_argument("--out", type=argparse.FileType('w'),
                        default=sys.stdout,
                        help="Output file. Defaults to stdout")
    args = parser.parse_args()

    if args.sing:
        logging.info("Singing")
        sing(args.out)
    else:
        logging.info("Not singing")

    logging.info("END: coke.py")

if __name__ == '__main__':
    main()
