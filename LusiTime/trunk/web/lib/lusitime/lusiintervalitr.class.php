<?php

namespace LusiTime;

require_once( 'lusitime.inc.php' );

/*
 * The class representing an iterator in an interval of time.
 */
class LusiIntervalItr {

    private $from  = null;
    private $to    = null;
    private $start = null;
    private $stop  = null;

    public function __construct($from,$to) {
        $this->from = $from;
        $this->to   = $to;
    }
    public function next_day() {

        // Next interval begins either from the 'from' limit passed to the constructor,
        // or from the previously calculated 'stop' time.
        //
        $this->start = is_null($this->start) ? $this->from : $this->stop;
        if( $this->start->to64() >= $this->to->to64()) return false;

        // And it ends either in +24 hours after the begin time of the interval,
        // or at the 'to' time passed to the constructor, whichever comes first.
        //
        $start_midnight = LusiTime::parse($this->start->toStringDay().' 00:00:00');
        $this->stop = new LusiTime($start_midnight->sec + 24 * 3600,  $start_midnight->nsec);
        //
        // Note that we also need to catch the Day Time Saving shift in +/-1 hr. In
        // that scenario adding exactly 24 hours would bring us either 1 hr short
        // or 1 hr beyond the desired day.
        //
        $stop_midnight = LusiTime::parse($this->stop->toStringDay().' 00:00:00');
        if($stop_midnight->to64() == $start_midnight->to64()) {

            // We're 1 hr short. Let's correct this by adding 25 hours.
            //
            $this->stop = new LusiTime($start_midnight->sec + 25 * 3600,  $start_midnight->nsec);

        } else {

            // this will automatically correct for 1 hr beyon the midnight
            // (if) added in an opposite direction.
            //
            $this->stop = $stop_midnight;
        }
        if( $this->to->to64() < $this->stop->to64()) $this->stop = $this->to;

        return true;
    }
    public function start() { return $this->start; }
    public function stop () { return $this->stop; }
}
?>
