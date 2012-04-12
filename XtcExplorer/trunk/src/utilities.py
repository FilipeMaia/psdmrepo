#-----------------------------------------
# PyanaOptions
#-----------------------------------------

class PyanaOptions( object ):
    """Class PyanaOptions

    Collection of functions to convert the string options of pyana
    into values, or lists of values, of the expected type
    """
    def __init__( self ):
        pass

    def getOptString(self, options_string) :
        """Return the string, strip of any whitespaces
        """
        if options_string is None:
            return None

        # make sure there are no newline characters here
        options_string = options_string.strip()

        if ( options_string == "" or
             options_string == "None" or
             options_string == "No" ) :
            return None

        # all other cases:
        return options_string


    def getOptStrings(self, options_string) :
        """Return a list of strings 
        """
        if options_string is None:
            return None

        # strip off any leading or trailing whitespaces
        options_string = options_string.strip()

        # make sure there are no newline characters here
        options_string = options_string.split("\n")
        options_string = " ".join(options_string)

        # make a list (split by whitespace)
        options = options_string.split()

        if len(options)==0 :
            return []

        elif len(options)==1 :
            if ( options_string == "" or
                 options_string == "None" or
                 options_string == "No" ) :
                return []

        # all other cases:
        return options

    def getOptStringsDict(self, options_string) :
        """Return a dictionary of strings
        """
        if options_string is None:
            return {}
        
        mylist = self.getOptStrings(options_string)
        mydict = {}
        for entry in mylist:
            items = entry.split(":")
            if len(items) > 1 :
                mydict[items[0]] = items[1].strip('([])').split(',')
            else:
                mydict[items[0]] = None
                
        return mydict

    def getOptStringsList(self, options_string) :
        """Return a list of strings (label) and VALUES
        Alternative to Dictionary, but it's sorted in whichever order was requested.
        """
        if options_string is None:
            return []
        
        mylist = self.getOptStrings(options_string)
        for entry in mylist:
            # location of this entry in list
            loc = mylist.index(entry)

            # split 
            items = entry.split(":")

            label = items[0]
            options = None

            if len(items) > 1 :
                options = items[1].strip('([])').split(',')

            # replace entry with tuple of (label, options)
            mylist[loc] = (label,options)
                
        return mylist

           
    def getOptIntegers(self, options_string):
        """Return a list of integers
        """
        if options_string is None: return None

        opt = self.getOptStrings(options_string)
        N = len(opt)
        if N is 1:
            return int(opt)
        if N > 1 :
            items = []
            for item in opt :
                items.append( int(item) )
            return items
            

    def getOptInteger(self, options_string):
        """Return a single integer
        """
        if options_string is None:   return None
        if options_string == "":     return None
        if options_string == "None": return None
        return int(options_string)


    def getOptBoolean(self, options_string):
        """Return a boolean
        """
        if options_string is None: return None

        opt = options_string
        if   opt == "False" or opt == "0" or opt == "No" or opt == "" : return False
        elif opt == "True" or opt == "1" or opt == "Yes" : return True
        else :
            print "utilities.getOptBoolean: cannot parse option ", opt
            return None

    def getOptBooleans(self, options_string):
        """Return a list of booleans
        """
        if options_string is None: return None

        opt_list = self.getOptStrings(options_string)
        N = len(opt_list)
        if N == 0 : return None
        
        opts = []
        for opt in optlist :
            opts.append( self.getOptBoolean(opt) )

        return opts


    def getOptFloats(self, options_string):
        """Return a list of integers
        """
        if options_string is None: return None

        opt = self.getOptStrings(options_string)
        N = len(opt)
        if N is 1:
            return float(opt)
        if N > 1 :
            items = []
            for item in opt :
                items.append( float(item) )
            return items
            

    def getOptFloat(self, options_string):
        """Return a single integer
        """
        if options_string is None: return None
        if options_string == "" : return None
        if options_string == "None" : return None

        return float(options_string)



#-----------------------------------------
# Data Storage Classes
#-----------------------------------------
import numpy as np

class BaseData( object ):
    """Base class for container objects storing event data
    in memory (as numpy arrays mainly). Useful for passing
    the data to e.g. ipython for further investigation
    """

    def __init__(self, name,type="BaseData"):
        self.name = name
        self.type = type
        
    def show( self ):
        itsme = "\n%s \n\t name = %s" % (self.type, self.name)
        for item in dir(self):
            if item.find('__')>=0 : continue
            attr = getattr(self,item)
            if attr is not None:
                if type(attr)==str:
                    print item, "(str) = ", attr
                elif type(attr)==np.ndarray:
                    print item, ": ndarray of dimension(s) ", attr.shape
                else:
                    print item, " = ", type(attr)
                    
    def show2( self ):
        itsme = "\n%s from %s :" % (self.type, self.name)
        myplottables = self.get_plottables()
        for key, array in myplottables.iteritems():
            itsme += "\n\t %s: \t %s   " % (key, array.shape)
        print itsme
        return
                    
    def get_plottables_base(self):
        plottables = {}
        for item in dir(self):
            if item.find('__')>=0 : continue
            attr = getattr(self,item)
            if attr is not None:
                if type(attr)==np.ndarray:
                    plottables[item] = attr
        return plottables
                                
    def get_plottables(self):
        return self.get_plottables_base()
                                
                                
                                
class BldData( BaseData ):
    """Beam-Line Data 
    """
    def __init__(self, name, type="BldData"):
        BaseData.__init__(self,name,type)
        self.time = None
        self.damage = None
        self.shots = None
        self.energy = None
        self.position = None
        self.angle = None
        self.charge = None
        self.fex_sum = None
        self.fex_channels = None
        self.raw_channels = None
        self.raw_channels_volt = None
        self.fex_position = None


class IpimbData( BaseData ):
    """Ipimb Data (from Intensity and Position monitoring boards)
    """
    def __init__( self, name, type="IpimbData" ):
        BaseData.__init__(self,name,type)
        self.gain_settings = None
        self.fex_sum = None
        self.fex_channels = None
        self.fex_position = None
        self.raw_channels = None
        self.raw_voltages = None



class EncoderData( BaseData ):
    """Encoder data
    """
    def __init__( self, name, type="EncoderData" ):
        BaseData.__init__(self,name,type)
        self.values = None



class WaveformData( BaseData ):
    """Waveform data from Acqiris digitizers
    """
    def __init__( self, name, type="WaveformData" ):
        BaseData.__init__(self,name,type)
        self.wf = None
        self.average = None
        self.ts = None
        self.counter = None
        self.channels = None
        self.stack = None

    def get_plottables(self):
        plottables = self.get_plottables_base()
        for ch in self.channels: 
            plottables["volt_vs_time_ch%d"%ch] = (self.ts[ch],self.wf[ch])
        return plottables

class EpicsData( BaseData ):
    """Control and Monitoring PVs from EPICS
    """
    def __init__( self, name, type="EpicsData" ):
        BaseData.__init__(self,name,type)
        self.values = None
        self.shotnr = None
        self.status = None
        self.severity = None



class ScanData( BaseData ) :
    """Scan data
    """
    def __init__(self, name, type="ScanData"):
        BaseData.__init__(self,name,type)

        self.scanvec = None
        self.arheader = None
        self.scandata = None



class ImageData( BaseData ):
    """Image data
    """
    def __init__(self, name, type="ImageData"):
        BaseData.__init__(self,name,type)
        self.image = None      # the image
        self.average = None    # the average collected so far
        self.maximum = None    # the max projection of images collected so far
        self.counter = 0       # nEvents in average
        self.dark = None       # the dark that was subtracted
        self.avgdark = None    # the average of accumulated darks
        self.ndark = 0         # counter for accumulated darks
        self.roi = None        # list of coordinates defining ROI
        
        # The following are 1D array if unbinned, 2D if binned (bin array being the first dim)
        self.spectrum = None   # Array of image intensities (1D, or 2D if binned)
        #self.projX = None      # Average image intensity projected onto horizontal axis
        #self.projY = None      # Average image intensity projected onto vertical axis
        self.showProj = False
        
        # The following are always 2D arrays, binned. bins vs. values
        self.projR = None      # Average image intensity projected onto radial axis (2D)
        self.projTheta = None  # Average image intensity projected onto polar angle axis (2D_
        
    def get_plottables(self):
        plottables = self.get_plottables_base()
        if self.roi is not None:
            try:
                c = self.roi
                print "image? ", self.image
                print "roi? ", self.image[c[0]:c[1],c[2]:c[3]]
                plottables["roi"] = self.image[c[0]:c[1],c[2]:c[3]]
            except:
                print "setting ROI failed, did you define the image? "
        return plottables
                


#-------------------------------------------------------
# Threshold  
#-------------------------------------------------------
class Threshold( object ) :
    """Class Threshold

    To keep track of threshold settings (value and area of interest)
    """
    def __init__( self, description = None ):
        """constructor
        @param description  
        """
        self.lower = None
        self.upper = None
        self.mask = None
        self.type = None
        self.region = None
        self.is_empty = False

        if description is None or description == "":
            self.is_empty = True
            return None
        
        print "setting up Threshold object based on description:", description
            
        words = description.split(' ')
        threshold = {}
        for w in words:
            n,d = w.split('=')
            threshold[n] = d

        print "Threshold:",
        if 'lower' in threshold:
            self.lower = float(threshold['lower'])
            print " lower = %.2f"% self.lower,
        if 'upper' in threshold:
            self.upper = float(threshold['upper']) 
            print " upper = %.2f"% self.upper,
        if 'mask' in threshold:
            self.mask = float(threshold['mask']) 
            print " mask = %.2f"% self.mask,
        if 'type' in threshold:
            self.type = threshold['type']
            print " type = %s"% self.type,
        if 'roi' in threshold:
            roi = [range.split(':') for range in threshold['roi'].strip('()').split(',')]
            self.region = [ int(roi[0][0]), int(roi[0][1]), int(roi[1][0]), int(roi[1][1]) ]
            print " region = %s"% self.region,
        print


            
            


