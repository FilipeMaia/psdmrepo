import numpy as np


print "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"
print "Importing pypsalg_cpp"
import pypsalg_cpp
print "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"



# Create test array
testarray = (np.random.rand(10)).astype(np.float32)
print "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"
print "Input test array", testarray
array_pointer, array_readfag = testarray.__array_interface__['data']
print "Data pointer 0x%0x"%array_pointer
print "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"


# Create instance of numpytest
#print "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"
#print "Creating instance of numpytest" 
#numpytest = pypsalg_cpp.numpytest()
#print "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"

# test C++ call to print array
#print "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"
#print "Calling C++ printarray(ndarray) function"
#numpytest.printArray(testarray)
#print "Input test array", testarray
#print "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"


# testing ndarray to numpy conversion
#print "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"
#print "Calling C++  ndarray outArray() function"
#testval = numpytest.outArray()
#print "contents of testval",testval
#print "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"


#print "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"
#print "numpy mean of testarray",np.mean(testarray)
#print "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"

#print "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"
#print "Calling C++  ndarray calcmean(ndarray)"
#mymean = numpytest.calcmean(testarray)
#print "my mean of testarray",mymean
#print "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"


#testarray2D = (np.random.rand(10,10)).astype(np.float32)
#numpytest.printArray2D(testarray2D,10,20)

#baseline_value = 0.0
#outarray =  numpytest.find_edges(testarray.astype(np.double),
#                                    baseline_value, 
#                                    0.1,
#                                    0.5,
#                                    0.0,
#                                    True)
#print outarray
#print baseline_value

#print "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"
#print "DELETING OUTGOING NUMPY ARRAYS"
#print "DELETING testval"
#del testval
#print "DELETING mymean"
#del mymean
#print "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"


waveform = np.random.rand(100)
print "waveform",waveform
#psalg = pypsalg_cpp.psalg()
edges = pypsalg_cpp.find_edges(waveform,
                               0.2,0.1,0.5,0.0,True)
print "edges",edges

#print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
#testarray = np.random.rand(10).astype(np.double)
#pypsalg_cpp.find_edges(testarray, 0.0, 0.0, 0.0, 0.0, True)
#print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

