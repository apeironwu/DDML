import sys, getopt

def main(argv):
    ## default values
    n_rnp = 100
    n_rds = 10
    try:
        opts, args = getopt.getopt(argv,"h",["n_rnp=","n_rds="])

    except getopt.GetoptError:
        print ('test.py --n_rnp <# replication> --n_rds <# data splitting>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print ('test.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("--n_rnp"):
            n_rnp = arg
        elif opt in ("--n_rds"):
            n_rds = arg
    
    print ('> n_rnp', n_rnp)
    print ('> n_rds', n_rds)

if __name__ == "__main__":
   main(sys.argv[1:])