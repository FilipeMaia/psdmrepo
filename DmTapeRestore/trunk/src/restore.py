
from datetime import datetime

from RegDB.experiment_info import name2id, id2name


def ns_to_date(ns_time):
    """ Return date string from ns-timestamp """
    return datetime.fromtimestamp(ns_time / 1E9)
    
def exper_id_name(exp_spec):
    """ Convert experiment spec to experiment name and id.
    Return as dict.
    """
    
    st = {}
    if exp_spec.isdigit():
        st['expid'] = int(exp_spec)
        st['expname'] = id2name(st['expid'])
    else:
        st['expname'] = exp_spec
        st['expid'] = name2id(st['expname'])


    if st['expname'] and st['expid']:
        return st
    else:
        raise ValueError("missing experiment info")
    
