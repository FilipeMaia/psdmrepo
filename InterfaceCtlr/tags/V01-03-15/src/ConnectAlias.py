


__alias_map = { 
    'dqm' : 'file:/reg/g/psdm/psdatmgr/ic/.icdb-dqm-conn',
    'mon' : 'file:/reg/g/psdm/psdatmgr/ic/.icdb-ffb-conn',
    'def' : 'file:/reg/g/psdm/psdatmgr/ic/.icdb-conn'
}


def connect_alias(conn_str):
    """ Return connection-str for an alias otherwise the 
    connection-str itself
    """

    return __alias_map.get(conn_str, conn_str)
