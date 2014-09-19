"""
Circular Buffer: A buffer of fixed length. Data is added to buffer
until buffer is full. Then first event is discarded and the new one
data is appended to end of circular buffer.
"""

# Import Standard PYTHON modules
import collections

# Import custom PYTHON modules



# Definition of the Circular Buffer class
class CircularBuffer : 
    """
    Circular Buffer class. Events appending to end of buffer but
    maintains same size by discarding old values    
    """

    def __init__(self, size) :
        """
        Constructor for Circular Buffer
        """

        print "Creating Circular Buffer of size ", size
        self.__buffer = collections.deque(maxlen=size)


    def add(self, data) :
        """
        Add data to circular buffer
        """
        self.__buffer.append(data)


    def data(self) :
        """
        Access to the buffer
        """
        return self.__buffer



# Code to test circular buffer code
if __name__ == "__main__" :

    # Create the circular buffer
    lastTenEvents = CircularBuffer(10)

    # loop over events, append data
    for evt in range(1,100) :
        lastTenEvents.add(evt)
        print "Buffer: ",lastTenEvents.data()
