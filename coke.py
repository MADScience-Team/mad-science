import sys, logging, argparse
##########
# Just writing a bit of Python code to make sure I have
# my satisfaction on this new macbook. Github working, etc.
##########


def main():

  parser = argparse.ArgumentParser()
  parser.add_argument("sing",action='store_true')
  args = parser.parse_args()
  
  song = """The sun will always shine.
The birds will always sing.
As long as there is thirst,
There's always the real thing.
"""

  if args.sing:
    print(song)

if __name__ == '__main__':
  main()
