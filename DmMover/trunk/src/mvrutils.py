

def network_name(hostname):
    """ Translate the daq name to the psana network name e.g.: daq-xpp-dss01 -> 10.1.1.1 """

    #if hostname.startswith('daq-cxi-'):
    #   return "10.1.1.%d" % int(hostname[-2:])
    return "10.1.1.1"
