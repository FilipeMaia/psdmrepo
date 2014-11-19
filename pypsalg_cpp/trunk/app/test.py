import psana
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


print "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"
print "TEST FINITE IMPULSE RESPONSE (2 argument)"
print "Random array for filter"
filter = np.random.rand(10)
sample = np.random.rand(100)
print "Filter Pointer 0x%0x"%(filter.__array_interface__['data'])[0]
print "Sample Pointer 0x%0x"%(sample.__array_interface__['data'])[0]
print "Calling FINITE IMPULSE RESPONSE"
output = pypsalg_cpp.finite_impulse_response(filter,sample)
print "Output Pointer 0x%0x"%(output.__array_interface__['data'])[0]
print "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"



print "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"
print "MOMENT TEST"

print "Running moment (3 argument)"
input = np.random.rand(10)
moment1 = pypsalg_cpp.moments(input,10.0,2.0)
print moment1

print "Running moment (4 argument)"
input2 = np.random.rand(10)
moment2 = pypsalg_cpp.moments(input,input2,10.0,2.0)
print moment2

print "MOMENT TEST 2D"
print "Running 2 argument"
input_2D = np.random.rand(10,10)
moment_2D_2 = pypsalg_cpp.moments(input_2D,3.0)
print moment_2D_2

print "MOMENT 2D BIT MASK"
rowmask = (np.random.rand(10)* 100).astype(np.uint32)
mask = (np.random.rand(10,10) * 100).astype(np.uint32)
moment_2D_mask = pypsalg_cpp.moments(input_2D,rowmask,mask,10.0)
print moment_2D_mask

print "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"



print "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"
print "FIND EDGES TEST"
waveform = np.random.rand(10)
edges = pypsalg_cpp.find_edges(waveform, 0.0, 0.1, 0.5, 0.0, True)
print edges
print "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"


#print "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"
#print "Find Peak Test"
#input = np.sin(np.arange(0.0,10*np.pi,0.1))
#peaks = pypsalg_cpp.find_peaks(input,0.1,10)
#print peaks
#print "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"


print "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"
print "Linear Fit Test"
input_test = np.random.rand(10)
pos = (np.random.rand(10) * 100).astype(np.uint32)
norm = 1.0
fit_result = pypsalg_cpp.line_fit(input_test,pos,norm)
print fit_result

print "Linear Fit 2 Test"
norm = np.random.rand(10)
fit_result = pypsalg_cpp.line_fit(input_test,pos,norm)
print fit_result
print "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"




print "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"
print "dist_rms test"
input_test = np.random.rand(10) * 100.0
norm = 1.0
baseline = np.random.rand(3)
rms_result = pypsalg_cpp.dist_rms(input_test, norm, baseline)
print rms_result

print "dist_rms_test_2"
norm_vec = np.random.rand(10)
rms_result = pypsalg_cpp.dist_rms(input_test, norm_vec, baseline)
print rms_result
print "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"



print "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"
print "dist_fwhm"
fwhm_result = pypsalg_cpp.dist_fwhm(input_test,norm,baseline)
print fwhm_result

print "dist_fwhm_2"
fwhm_result = pypsalg_cpp.dist_fwhm(input_test,norm_vec,baseline)
print fwhm_result
print "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"



print "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"
print "parab_interp"
parab_result = pypsalg_cpp.parab_interp(input_test, norm, baseline)
print parab_result

print "parab_interp 2"
parab_result = pypsalg_cpp.parab_interp(input_test, norm_vec, baseline)
print parab_result
print "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"


print "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"
print "parab_fit"
parab_fit_result = pypsalg_cpp.parab_fit(input_test)
print parab_fit_result

print "parab fit 2"
parab_fit_result = pypsalg_cpp.parab_fit(input_test,3,0.5)
print parab_fit_result
print "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"



print "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP"
print "common mode"
test_data = (np.random.rand(100) * 100).astype(np.int32)
test_baseline = (np.random.rand(100)*10)

commonmode_data = pypsalg_cpp.commonmode_lroe(test_data,test_baseline)
print commonmode_data
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


#waveform = np.random.rand(100)
#print "waveform",waveform
#psalg = pypsalg_cpp.psalg()
#edges = pypsalg_cpp.find_edges(waveform,
#                               0.2,0.1,0.5,0.0,True)
#print "edges",edges

#print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
#testarray = np.random.rand(10).astype(np.double)
#pypsalg_cpp.find_edges(testarray, 0.0, 0.0, 0.0, 0.0, True)
#print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

