"""
Part Finder PartGNN runner
TAH Jan 2021
"""

from partfind.partgnn import PartGNN
from partfind.parameter_parser import par_parser
import argparse



def main():
    """
    Parse command line params
    """
    args = par_parser()
    model = PartGNN(args)

    
    
if __name__ == "__main__":
    # execute only if run as a script
    main()