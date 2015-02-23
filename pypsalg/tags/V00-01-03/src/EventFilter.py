"""
General class used to filter incoming data by some value and then
processes the events. 

For example, updating updating radial integral plots for different
laser-xray delays.
"""

# Import Standard PYTHON modules
import numpy as np


# Import custom PYTHON modules



# Definition of the EventFilter class
class EventFilter :
    """
    The EventFilter class is designed to set up an array of analyses that
    get updated given a certain value. 
    """

    # Constructor
    def __init__(self, start, end, bins, updateMethod, plotObj, *args, **kwargs):
        """
        Constructor for EventFilter
          - plotObj ==> 
        """
    
        print "Creating EventFilter"
        
        # Data members --> make all private for now
        self.__plotList = []     # List of plots, each element will be
                                  # reference to a plot
        
        self.__binEdges = []     # List of bin edges

        # reference to the name of the class method called to update
        # every binned event
        self.__updateMethod = updateMethod
        
        # Now call 'create list' to the object list & initialise
        self.__createList(start, end, bins, plotObj, *args, **kwargs)

        
    
    def __createList(self, start, end, bins, plotObj, *args, **kwargs):
        """
        Create list of value bins and list of plots
        """

        print "Creating Bin List"
        
        # create bin edges
        self.__binEdges = np.linspace(start, end, bins+1)
        
        # Initialise an empty list of size bins
        Objlist = [ None ] * bins

        # Fill list with copies of plotObj initialised by *args & **kwargs 
        self.__plotList = [plotObj(*args, **kwargs) for Obj in Objlist]
               
        # prepend an underflow bin
        self.__plotList.insert(0, plotObj(*args, **kwargs))
        
        # append an overflow bin
        self.__plotList.append(plotObj(*args, **kwargs))

        
#        print "BinEdges: ", self.__binEdges
#        print "PlotList: ", self.__plotList



#    def update(self, value, data, *args, **kwargs):
    def update(self, value, data, **kwargs):
        """
        Use the 'value' to update the appropiate plot
        """

        #        print "update plot called"
        
        # Find bin
        bin = self.__findBin(value)
        #print "Bin Number: ",bin
        #        (self.__plotList[bin]).append(value)    
        #        (self.__plotList[bin]) += data

        # Call the update method
#        getattr(self.__plotList[bin], self.__updateMethod)(data,
#                                                            *args, **kwargs)
        
        getattr(self.__plotList[bin], self.__updateMethod)(data, **kwargs)
        

        
    def __findBin(self,value):
        """
        Method to find the bin given value
        """

        # At the moment, a simple linear search is used to find the
        # bin.  But by construction, the bins are ordered, which means
        # faster search algorithms can be used. Eg:- Binary Search.

#        print "Find Bin Called"
#        print "Value: ", value
        
        # Check for under/overflow
        if value < self.__binEdges[0] :
            return 0
        if value > self.__binEdges[-1] :
            return -1
        
        # Not underflow nor overflow - now search bins
        for index in range(0,len(self.__binEdges)-1) :
            if value>=self.__binEdges[index] and value<self.__binEdges[index+1]:
                return index+1
        
        # Got here ---> must be an error
        print "ERROR NO BIN FOUND !!!"
        pass
    
    
    
    def list(self) :
        """
        Accessor to internal plot list
        """
        return self.__plotList
        
    

# Code to test EventFilter code
if __name__ == "__main__" :
    
    from AnalysisPipeline import *

    
    # Create an array that'll be our store
    #    dataArray = []

    # Create an EventFilter Object
    binByValue = EventFilter(start = -1.0,
                            end = 1.0,
                            bins = 5,
                            plotObj=AnalysisPipeline,ID=50,
                            updateMethod="update" )

    
    # Create the BinByBin list
    #    binByValue.createList(-1.0,1.0,5, dataArray, "append")

    # loop over 100 events
    for evtNum in xrange(0,100) :

        # generate a random number
        delay  = np.random.rand()
        data   = np.random.rand(1024,1024)
        print "evtNum:",evtNum," delay:",delay
        
        # Use random number, bin and store
        binByValue.update(delay, data, gmd=1.0)
    

        
    # print out the internal plot list
    print binByValue.list()
    
    
