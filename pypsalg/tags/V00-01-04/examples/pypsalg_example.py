import numpy as np
import pypsalg
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create test array
testarray = (np.random.rand(10)).astype(np.float32)
logger.info("Input test array %s"%testarray)

# For debugging, get the pointer to the array data
array_pointer, array_readfag = testarray.__array_interface__['data']
logger.debug("Data pointer 0x%0x"%array_pointer)




logger.info("TEST FINITE IMPULSE RESPONSE (2 argument)")

# Create an array and print out pointers to array data 
logger.debug("Random array for filter")
filter = np.random.rand(10)
sample = np.random.rand(100)
logger.debug("Filter Pointer 0x%0x"%(filter.__array_interface__['data'])[0])
logger.debug("Sample Pointer 0x%0x"%(sample.__array_interface__['data'])[0])


logger.info("Calling FINITE IMPULSE RESPONSE")
output = pypsalg.finite_impulse_response(filter,sample)
logger.debug("Output Pointer 0x%0x"%(output.__array_interface__['data'])[0])





logger.info("MOMENT TEST")


logger.info("Running moment (3 argument)")
input_array = np.random.rand(10).astype(np.float64)
moment1 = pypsalg.moments_1D(input_array,10.0,2.0)
logger.info("%s"%moment1)

logger.info("Running moment (4 argument)")
input2 = np.random.rand(10)
moment2 = pypsalg.moments_1D(input_array,input2,10.0,2.0)
logger.info("%s"%moment2)

logger.info("MOMENT TEST 2D")
logger.info("Running 2 argument")
input_2D = np.random.rand(10,10)
moment_2D_2 = pypsalg.moments_2D(input_2D,3.0)
logger.info("%s"%moment_2D_2)

logger.info("MOMENT 2D BIT MASK")
rowmask = (np.random.rand(10)* 100).astype(np.uint32)
mask = (np.random.rand(10,10) * 100).astype(np.uint32)
moment_2D_mask = pypsalg.moments_2D(input_2D,rowmask,mask,10.0)
logger.info("%s"%moment_2D_mask)




logger.info("FIND EDGES TEST")
waveform = np.random.rand(10)
edges = pypsalg.find_edges(waveform, 0.0, 0.1, 0.5, 0.0, True)
logger.info("%s"%edges)




logger.info("Find Peak Test")
input = np.sin(np.arange(0.0,10*np.pi,0.1))
peaks = pypsalg.find_peaks(input,0.1,10)
logger.info("%s"%peaks)




logger.info("Linear Fit Test")
input_test = np.random.rand(10)
pos = (np.random.rand(10) * 100).astype(np.uint32)
norm = 1.0
fit_result = pypsalg.line_fit(input_test,pos,norm)
logger.info("%s"%fit_result)

logger.info("Linear Fit 2 Test")
norm = np.random.rand(10)
fit_result = pypsalg.line_fit(input_test,pos,norm)
logger.info("%s"%fit_result)






logger.info("dist_rms test")
input_test = np.random.rand(10) * 100.0
norm = 1.0
baseline = np.random.rand(3)
rms_result = pypsalg.dist_rms(input_test, norm, baseline)
logger.info("%s"%rms_result)

logger.info("dist_rms_test_2")
norm_vec = np.random.rand(10)
rms_result = pypsalg.dist_rms(input_test, norm_vec, baseline)
logger.info("%s"%rms_result)





logger.info("dist_fwhm")
fwhm_result = pypsalg.dist_fwhm(input_test,norm,baseline)
logger.info("%s"%fwhm_result)

logger.info("dist_fwhm_2")
fwhm_result = pypsalg.dist_fwhm(input_test,norm_vec,baseline)
logger.info("%s"%fwhm_result)





logger.info("parab_interp")
parab_result = pypsalg.parab_interp(input_test, norm, baseline)
logger.info("%s"%parab_result)

logger.info("parab_interp 2")
parab_result = pypsalg.parab_interp(input_test, norm_vec, baseline)
logger.info("%s"%parab_result)




logger.info("parab_fit")
parab_fit_result = pypsalg.parab_fit(input_test)
logger.info(parab_fit_result)

logger.info("parab fit 2")
parab_fit_result = pypsalg.parab_fit(input_test,3,0.5)
logger.info("%s"%parab_fit_result)





logger.info("common mode")
test_data = (np.random.rand(100) * 100).astype(np.int32)
test_baseline = (np.random.rand(100)*10)

commonmode_data = pypsalg.commonmode_lroe(test_data,test_baseline)
logger.info("%s"%commonmode_data)



logger.info("Creating random image for hit finder testing")
test_image = (np.random.rand(10,10) * 100).astype(np.uint32)
print test_image

hit_count = pypsalg.count_hits(test_image, 90)
print hit_count
            
sum_hit = pypsalg.sum_hits(test_image, 70,10)
print sum_hit

count_excess = pypsalg.count_excess(test_image, 50)
print count_excess

sum_excess = pypsalg.sum_excess(test_image, 75, 10)
print sum_excess



logger.info("Testing rolling average")
test_data = np.ones(10).astype(np.float64) \
            +  ((np.random.rand(10).astype(np.float64)-0.5) * 0.2)
print test_data


avg = test_data

for i in range(10) :
    test_data = np.ones(10).astype(np.float64) \
                + ((np.random.rand(10).astype(np.float64)-0.5) * 0.2)
    avg = pypsalg.rolling_average(test_data,avg, 0.90)
    print avg

