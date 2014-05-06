#!/reg/common/package/python/2.7.5/x86_64-rhel5-gcc41-opt/bin/python
#
import numpy
from scipy.linalg import toeplitz
from scipy.linalg import inv

if __name__ == "__main__":

    nwts = 50
    
    acf = numpy.array([0.645853,0.616422,0.616519,0.615200,0.614732,0.614930,0.616235,0.615859,0.615520,0.615483,0.615558,0.615714,0.614637,0.613582,0.613542,0.613407,0.613672,0.612058,0.614098,0.611701,0.613373,0.612557,0.611157,0.611631,0.613428,0.609893,0.611193,0.611205,0.611876,0.609742,0.609356,0.610410,0.608913,0.609330,0.609149,0.607156,0.606992,0.606695,0.606515,0.605970,0.604174,0.604028,0.604753,0.603046,0.603870,0.602358,0.602934,0.602722,0.600472,0.599776][:nwts],numpy.float)
    print 'acf[] ',acf

    signal = numpy.array([1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,0.99,0.96,0.93,0.90,0.87,0.84,0.81,0.78,0.75,0.72,0.69,0.66,0.63,0.60,0.57,0.54,0.51,0.48,0.45,0.42,0.39,0.36,0.33,0.30,0.27,0.24,0.21,0.18,0.15,0.12,0.09,0.06,0.03,0.01,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00][na:na+nwts],numpy.float)
    print 'signal[] ',signal
    
    acfm = toeplitz(acf)
    print 'acfm[] ',acfm[:3,:3]

    acfminv = inv(acfm)
    print 'acfi[] ',acfminv[:3,:3]

    w = numpy.dot(acfminv,signal)
    wli = []
    for i in range(len(w)):
        wli.append(w[len(w)-i-1])

    weights = numpy.array(wli)
    norm = numpy.dot(weights,signal)
    print 'norm ',norm

    weights = weights*1.00/norm
    w = w*1.00/norm
    print 'weights ',weights

    norm = numpy.dot(weights,signal)
    print 'norm ',norm

    print 'weights ',w


