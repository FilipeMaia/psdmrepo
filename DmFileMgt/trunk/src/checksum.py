

import logging
import subprocess
import threading


def md5_worker(fn, result_dict):
    """ Calculate checksum """
    proc = subprocess.Popen(["md5sum", fn], stdout=subprocess.PIPE,  close_fds=True)
    res = proc.communicate()[0]
    print "done ", fn, len(res)
    try:
        result_dict['md5'] =  res.split()[0]
    except IndexError:
        result_dict['md5'] =  '-1'
    result_dict['fn'] =  fn
    
def cmp_md5(fn, fn_cmp):
    """ compare md5sum for two files, use threads to run in parallel """
    threads = []
    results = []
    for fname in (fn, fn_cmp):
        rd = {}
        results.append(rd)
        t = threading.Thread(target=md5_worker, args=(fname,rd))
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
        
    logging.info("Checksum %s %s %s,", results[0]["md5"], results[1]["md5"], fn)
    return results[0]["md5"] == results[1]["md5"]

