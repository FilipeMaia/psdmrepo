<?php

namespace LusiTime;

require_once( 'lusitime.inc.php' );

/*
 * The class representing an interval of time in Web applications. It has
 * two data members representing the begin and end times of the interval.
 * The end time is allowed to be null which would mean an open-ended
 * interval. The bgin time must always be provided.
 */
class LusiInterval {

    /* Data members
    */
    public $begin;
    public $end;
    private static $dayspermonth = array( 1 => 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 );

    /**
     * Corrected number of days in a month of a year. The function will take
     * into account leap years.
     *
     * @param integer $month
     * @param integer $year
     * @return integer
     */
    private static function daysInThisMonth( $month, $year ) {
        if( $month == 2 && ( 0 == ( $year - 1972 ) % 4 )) return 29;
        return LusiInterval::$dayspermonth[$month];
    }

    /* Constructor
     */
    public function __construct($begin, $end=null) {
        if( !is_null($end) && ( $begin->greaterOrEqual( $end )))
            throw new LusiTimeException(
                __METHOD__, "the begin time ".$begin->toStringShort()." must be strictly less than the end time ".$end );

        $this->begin = $begin;
        $this->end = $end;
    }
    public function toSeconds() {
        return $this->end->sec - $this->begin->sec;
    }
    public function toMinutes() {
        $total_sec = $this->end->sec - $this->begin->sec;
        return intval( $total_sec / 60);
    }
    public function toStringHM() {
        $total_sec = $this->end->sec - $this->begin->sec;
        $hr  = intval( $total_sec / 3600);
        $min = intval(($total_sec % 3600) / 60);
        $sec =        ($total_sec % 3600) % 60;
        return sprintf("%02d:%02d", $hr, $min);
    }

    /**
     * Return an array of subintervals (of the current class) each representing
     * a day. The first day will begin at the begin time of the current interval,
     * and the last day will end with the end time of the current interval.
     * If the interval is open-ended then the result will contain the current
     * interval only.
     *
     * @return LusiInterval
     */
    public function splitIntoDays() {
        if( is_null( $this->end )) return array( $this );

        /* The algorithm will iterate over a sequence of triplets
         *
         *   [year][month][day]
         *
         * to yeald all days within the interval
         */
        $first = LusiInterval::time2triplet( $this->begin );
        $first_time = $this->begin;

        $last   = LusiInterval::time2triplet( $this->end );
        $last_time = $this->end;

        $prev = $first;
        $prev_time = $first_time;

        $result = array();
        while( !LusiInterval::triplets_are_equal( $prev, $last )) {
            $next = LusiInterval::next_triplet( $prev );
            $next_time = LusiInterval::triplet2time( $next );
            array_push( $result, new LusiInterval( $prev_time, $next_time ));
            $prev = $next;
            $prev_time = $next_time;
        }
        array_push( $result, new LusiInterval( $prev_time, $last_time ));
        return $result;
    }

    private static function triplets_are_equal( $lhs, $rhs ) {
        return ( $lhs[0] == $rhs[0] ) && ( $lhs[1] == $rhs[1] ) && ( $lhs[2] == $rhs[2] );
    }
    private static function next_triplet( $ymd ) {
        $y = $ymd[0];
        $m = $ymd[1];
        $d = $ymd[2];
        if( $d == LusiInterval::daysInThisMonth( $m, $y )) {
            $d = 1;
            if( $m == 12 ) {
                $m = 1;
                $y++;
            } else {
                $m++;
            }
        } else {
            $d++;
        }
        $triplet = array( $y, $m, $d );
        return $triplet;
    }
    private static function time2triplet( $time ) {
        $ymd = explode( '-', $time->toStringDay());
        return array((int)$ymd[0], (int)$ymd[1], (int)$ymd[2]);
    }
    private static function triplet2time( $ymd ) {
        //                           h  m  s  month    day      year
        return new LusiTime( mktime( 0, 0, 0, $ymd[1], $ymd[2], $ymd[0] ));
    }
}

/* The unit test for the class follows below. Uncomment it to use it.
 *
$begin = LusiTime::parse( "2008-02-27 19:24:00" );
echo '<br>'.$begin;
$end = LusiTime::parse( "2009-08-10 08:00:00" );
echo '<br>'.$end;
$i = new LusiInterval( $begin, $end );
$days = $i->splitIntoDays();
foreach( $days as $d ) {
  echo "<br>[ ".$d->begin." : ".$d->end." )";
}
*
*/
?>

