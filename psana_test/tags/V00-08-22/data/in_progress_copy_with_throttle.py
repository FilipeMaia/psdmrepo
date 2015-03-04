import argparse

import psana_test.liveModeSimLib as liveModeLib

programDescription='''
driver program to copy a file from src to dest with delay and throttling. For simulating live mode.
'''

programDescriptionEpilog='''
'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=programDescription, 
                                     epilog=programDescriptionEpilog,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument('--inprogress_ext',type=str, help="extension while file is in progress", default=".inprogress")
    parser.add_argument('--start_delay',type=float, help="seconds to wait before starting the copy", default=0.0)
    parser.add_argument('--mb_per_write',type=float, help="number of megabytes to write with each copy", default=1.0)
    parser.add_argument('--delay_between_writes',type=float, help="seconds to sleep between writes", default=.2)
    parser.add_argument('--max_mbs',type=float, help="maximum number of megabytes to copy. <=0 means all.", default=-1.0)
    parser.add_argument('--force',action='store_true', help="force overwrite of dest", default=False)
    parser.add_argument('-v','--verbose',action='store_true', help="verbose", default=False)
    parser.add_argument('src', type=str, help="source file")
    parser.add_argument('dest', type=str, help="destination file")
    
    args = parser.parse_args()
    liveModeLib.inProgressCopyWithThrottle(args.src, args.dest, args.start_delay, args.mb_per_write, args.inprogress_ext,
                                           args.max_mbs, args.delay_between_writes, args.force, args.verbose)

    
