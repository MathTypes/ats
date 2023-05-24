import sys, getopt
import os

def main(argv):
    inputfile = ''
    outputfile = ''
    inputfile = str(sys.argv)
    inputfile = " ".join(sys.argv[1:])
    inputfile = inputfile.replace("^", " ")
    outputfile = inputfile.replace(" ", "_")
    print ('Input file is ', inputfile)
    print ('Output file is ', outputfile)
    os.rename(inputfile, outputfile)

if __name__ == "__main__":
   main(sys.argv[1:])