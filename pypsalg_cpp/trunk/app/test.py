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
print "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"
print "Creating instance of numpytest" 
numpytest = pypsalg_cpp.numpytest()
print "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"

# test C++ call to print array
print "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"
print "Calling C++ printarray(ndarray) function"
numpytest.printArray(testarray)
print "Input test array", testarray
print "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"


# testing ndarray to numpy conversion
print "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"
print "Calling C++  ndarray outArray() function"
testval = numpytest.outArray()
print "contents of testval",testval
print "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"


print "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"
print "numpy mean of testarray",np.mean(testarray)
print "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"

print "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"
print "Calling C++  ndarray calcmean(ndarray)"
mymean = numpytest.calcmean(testarray)
print "my mean of testarray",mymean
print "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"


#print "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"
#print "DELETING OUTGOING NUMPY ARRAYS"
#print "DELETING testval"
#del testval
#print "DELETING mymean"
#del mymean
#print "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"




