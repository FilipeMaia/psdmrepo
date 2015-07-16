/**
 * Module with static definitions used throughout the application
 * 
 */
define ([] ,

function () {

    /**
     * Server-side processing options
     *
     * @type Array
     */
    var _PROCESSING_OPTIONS = [

        {   operator: "raw" ,
            desc: "Returns unprocessed data"} ,

        {   operator: "firstSample" ,
            desc:     "Returns the first sample in a bin. This is the default sparsification operator."} ,

        {   operator: "lastSample" ,
            desc:     "Returns the last sample in a bin."} ,

        {   operator: "firstFill" ,
            desc:     "Similar to the firstSample operator with the exception that we alter the timestamp " + 
                      "to the middle of the bin and copy over the previous bin's value if a bin does not have any samples."} ,

        {   operator: "lastFill" ,
            desc:     "Similar to the firstFill operator with the exception that we use the last sample in the bin."} ,

        {   operator: "mean" ,
            desc:     "Returns the average value of a bin. This is computed using SummaryStatistics " +
                      "and is SummaryStatistics.getMean()"} ,

        {   operator: "errorbar" ,
            desc:     "Returns the average value of a bin along with its standard deviation"} ,

        {   operator: "min" ,
            desc:     "Returns the minimum value in a bin. This is computed using SummaryStatistics " +
                      "and is SummaryStatistics.getMin()"} ,

        {   operator: "max" ,
            desc:     "Returns the maximum value in a bin. This is computed using SummaryStatistics " +
                      "and is SummaryStatistics.getMax()"} ,

        {   operator: "count" ,
            desc:     "Returns the number of samples in a bin. This is computed using SummaryStatistics " +
                      "and is SummaryStatistics.getN()"} ,

        {   operator: "ncount" ,
            desc:     "Returns the total number of samples in a selected time span."} ,

        {   operator: "nth" ,
            desc:     "Returns every n-th value."} ,

        {   operator: "median" ,
            desc:     "Returns the median value of a bin. This is computed using DescriptiveStatistics " +
                      "and is DescriptiveStatistics.getPercentile(50)"} ,

        {   operator: "std" ,
            desc:     "Returns the standard deviation of a bin. This is computed using SummaryStatistics " +
                      "and is SummaryStatistics.getStandardDeviation()"} ,

        {   operator: "jitter" ,
            desc:     "Returns the jitter (the standard deviation divided by the mean) of a bin. This is computed " +
                      "using SummaryStatistics and is SummaryStatistics.getStandardDeviation()/SummaryStatistics.getMean()"} ,

        {   operator: "ignoreflyer" ,
            desc:     "Ignores data that is more than the specified amount of std deviation from the mean in the bin. " +
                      "This is computed using SummaryStatistics. It takes two arguments, the binning interval and " +
                      "the number of standard deviations (by default, 3.0). It filters the data and returns onlythose values " +
                      "which satisfy Math.abs(val - SummaryStatistics.getMean()) <= numDeviations*SummaryStatistics.getStandardDeviation()"} ,

        {   operator: "flyers" ,
            desc:     "Opposite of ignoreflyers - only returns data that is more than the specified " +
                      "amount of std deviation from the mean in the bin."} ,

        {   operator: "variance" ,
            desc:     "Returns the variance of a bin. This is computed using SummaryStatistics " +
                      "and is SummaryStatistics.getVariance()"} ,

        {   operator: "popvariance" ,
            desc:     "Returns the population variance of a bin. This is computed using SummaryStatistics " +
                      "and is SummaryStatistics.getPopulationVariance()"} ,

        {   operator: "kurtosis" ,
            desc:     "Returns the kurtosis of a bin - Kurtosis is a measure of the peakedness. " +
                      "This is computed using DescriptiveStatistics and is DescriptiveStatistics.getKurtosis()"} ,

        {   operator: "skewness" ,
            desc:     "Returns the skewness of a bin - Skewness is a measure of the asymmetry. " +
                      "This is computed using DescriptiveStatistics and is DescriptiveStatistics.getSkewness()"}
    ] ;

    return {
        PROCESSING_OPTIONS: _PROCESSING_OPTIONS
    } ;
}) ;