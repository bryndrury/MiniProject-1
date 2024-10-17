import sys
from LebwohlLasher import LebwohlLasher

if __name__ == '__main__':
    if int(len(sys.argv)) == 5:
        PROGNAME = sys.argv[0]
        ITERATIONS = int(sys.argv[1])
        SIZE = int(sys.argv[2])
        TEMPERATURE = float(sys.argv[3])
        THREADCOUNT = int(sys.argv[4])
        
        LebwohlLasher(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, THREADCOUNT)
    elif int(len(sys.argv)) == 2:
        if sys.argv[1] == "--help":
            print("\nUsage: python3 {} <ITERATIONS> <SIZE> <TEMPERATURE> <THREADCOUNT>".format(sys.argv[0]))
            print("\nExample: python3 {} 1000 100 1.0 4".format(sys.argv[0]))
            print("\nIt is not recommended to use a size greater than 5000 as runtime will be extremly long .\n")
    else:
        print("Usage: python {} <ITERATIONS> <SIZE> <TEMPERATURE> <THREADCOUNT>".format(sys.argv[0]))
#=======================================================================

    
    