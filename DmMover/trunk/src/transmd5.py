
import os
import sys
import subprocess

path = os.path

def read_bbcp_checksum(lfn, anapath=True,chkext='md5'):
    """ read checksum from bbcp checksum output file 
    
    Return checksum-value, checksum-type, file-size
    """
    
    prefix = "/reg/d/psdm" if anapath else "/reg/d/ffb"
    
    sp = path.split(lfn)
    chkfile = path.join(prefix, sp[0].lstrip('/'), 'md5', sp[1]) + '.' + chkext

    # size for xtc file
    pfn = path.join(prefix, lfn.lstrip('/'))
    try:
        size = os.path.getsize(pfn)
    except OSError:
        size = -1

    if not path.exists(chkfile):
        #print "Missing", chkfile
        return None,None, size

    with open(chkfile) as fp:
        line = fp.readline()
    tok = line.split()
    try:
        chk_type, chk_value = tok[1], tok[2]
    except IndexError:
        # print "Error reading", chkfile, line.lstrip()
        chk_type, chk_value = None, None
    return  chk_value, chk_type, size


def comp_bbcp_checksum(lfn, verbose=False, pfn_prefix=None):
    """ Comapre checksums for the ffb and ana file copy using 
    the md5 files that were created by the transfers.  
    
    In addition compare the file sizes and optionally (pfn_prefix is set)
    calculate the checksum for one file copies and compare to the transfer
    checksum. 

    lfn = /<instr>/<exper>/<xtc>/<fn>
    """

    ana_chksum, ana_cktype, ana_size = read_bbcp_checksum(lfn)
    ffb_chksum, ffb_fcktype, ffb_size = read_bbcp_checksum(lfn, False)
        
    disk_comp = True
    if pfn_prefix:
        pfn = path.join(pfn_prefix, lfn.lstrip('/'))
        if os.path.exists(pfn):
            res = subprocess.check_output(["md5sum", pfn])
            md5_dsk = res.split()[0]
        if md5_dsk != ana_chksum:
            print "Mismatch disk", lfn
            disk_comp = False

    st_size = (ana_size == ffb_size) and ana_size >= 0
    st = ana_chksum == ffb_chksum
    if verbose:
        print "status: cksum={} size={} disk={} calcMd5={} Values: {} {}, {} {} {}".format(
            ana_chksum == ffb_chksum, st_size, disk_comp, pfn_prefix != None,
            ana_chksum, ffb_chksum, ana_size, ffb_size, lfn)
        
    return (ana_chksum == ffb_chksum) and disk_comp and st_size
