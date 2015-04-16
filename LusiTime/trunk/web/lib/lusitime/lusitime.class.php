<?php

namespace LusiTime;

require_once( 'lusitime.inc.php' );

/* Set the default timezone to prevent complains from PHP run time.
 *
 * NOTE: this parameter would need to be properly customized
 *       in order to use this software in a different location.
 */
date_default_timezone_set( 'America/Los_Angeles' );

/*
 * The class representing time in Web applications. It has
 * two data members representing the number of seconds since UNIX
 * epoch (GMT timezone) and the number of nanoseconds. The number
 * of nanoseconds is in the range of 0..999999999.
 */
class LusiTime {

    /* Data members
    */
    public $sec;
    public $nsec;

    /* Factory method for getting the current time
     */
    public static function now() {
        return new LusiTime( time()); }

    /* Factory method for getting the current time with the micro-seconds resolution
     * (if the host operating system permits).
     */
    public static function now_micro() {
        $now = gettimeofday();
        return new LusiTime( $now['sec'], 1000*$now['usec']);
    }

    /* Factory method for getting the time -1 hour from now
     */
    public static function minus_hour() {
        return new LusiTime( time() - 3600 ); }

    /* Factory method for getting the time -24 hours from now
     */
    public static function minus_day() {
        return new LusiTime( time() - 24*3600 ); }

    /* Factory method for getting the time as of today at 00:00:00
     */
    public static function today() {
        return LusiTime::parse( LusiTime::now()->toStringDay().' 00:00:00' ); }

    /* Factory method for getting the time as of yesterday at 00:00:00
     */
    public static function yesterday() {
        return LusiTime::parse( LusiTime::minus_day()->toStringDay().' 00:00:00' ); }

    /* Factory method for getting the time -1 week from now
     */
    public static function minus_week() {
        return new LusiTime( time() - 7*24*3600 ); }

    /* Factory method for getting the time -1 month from now
     *
     * NOTE: 1 "month" is equal to 31 day in this context
     */
    public static function minus_month() {
        return new LusiTime( time() - 31*24*3600 ); }

    /* Factory method from an input string to be parsed into an object
     * representing a timestamp in the local timezone.
     *
     * Input formats:
     *
     *   2009-05-19
     *   2009-05-19 17:59:49
     *   2009-05-19 12:18:49.123456789
     *   2009-05-19T17:59:49
     *   2009-05-19T12:18:49.123456789
     *
     * The last example illustrates how to specify nanoseconds. The number
     * of nanoseconds (if not empty) must be in the range of 0..999999999.
     *
     * The method will return an object of the class or it will
     * return null in case of any error.
     */
    public static function parse($str) {

        $expr = '/(\d{4})-(\d{1,2})-(\d{1,2})([ T]+(\d{1,2}):(\d{1,2}):(\d{1,2})(\.(\d{1,9}))?)?/';
        if( !preg_match( $expr, $str, $matches2 )) return null;

        // Begin with interpreting result as a time w/ optonal nanoseconds
        // in the default timezone. Normally the default timezone is the local one.
        //
        $year          = $matches2[1];
        $month         = $matches2[2];
        $day           = $matches2[3];
        $hour          = '00';
        $min           = '00';
        $sec           = '00';
        if(        isset($matches2[4])) {
            $hour      = $matches2[5];
            $min       = $matches2[6];
            $sec       = $matches2[7];
        }
        $default_tz_time_sec = strtotime( "{$year}-{$month}-{$day} {$hour}:{$min}:{$sec}" );
        if( !$default_tz_time_sec ) return null;

        $nsec = isset( $matches2[9] ) ? $matches2[9] : 0;

        return new LusiTime($default_tz_time_sec,$nsec);
    }

    /* Constructor
     */
    public function __construct($sec, $nsec=0) {
        if( $nsec < 0 or $nsec > 999999999 )
            throw new LusiTimeException(
                __METHOD__, "the number of nanoseconds ".$nsec." isn't in allowed range" );

        $this->sec = $sec;
        $this->nsec = $nsec;
    }

    /* Return a human-readable ISO representation
     * of date and time.
     */
    public function __toString() {
        return date("Y-m-d H:i:s", $this->sec).sprintf(".%09u", $this->nsec).date("O", $this->sec); }

    /* Return the data and time in 24 hours from the ones encoded in
     * the current object.
     */
    public function  in24hours () {
        return new LusiTime($this->sec + 24 * 3600) ; }

    /* Return the data and time in 25 hours from the ones encoded in
     * the current object.
     * 
     * NOTE: This function can be used to make proper adjustments due to Dayligh Savings
     *       time switch.
     */
    public function  in25hours () {
        return new LusiTime($this->sec + 25 * 3600) ; }

    /* Unlike the previous method this one would return a short (no
     * nanoseconds and time-zone) representation (ISO) of a human-readable
     * date and time.
     */
    public function toStringShort() {
        return date("Y-m-d H:i:s", $this->sec); }

    /* Return a human-readable ISO representation
     * of date and time (no nanoseconds) and UTC timezone:
     * 
     * 2014-06-03T13:47:30+00:00
     * 
     */
    public function toStringShortISO() {
        return date("Y-m-d", $this->sec).'T'.date("H:i:s", $this->sec).date("P", $this->sec); }

    /* Unlike the previous methods this one would return just a day
     * (yer-month-day) part of the timestamp in the human-readable
     * format.
     */
    public function toStringDay() {
        return date("Y-m-d", $this->sec); }

    /* Return a day(yer-month-day) part of the timestamp in the human-readable
     * format:
     * 
     *   20-FEB-12
     */
    public function toStringDay_1() {
        return date("d", $this->sec).'-'.strtoupper(date("M", $this->sec)).'-'.date("y", $this->sec); }

    /* Unlike the previous methods this one would return just the hours-minutes-seconds
     * part of the timestamp in the human-readable format.
     */
    public function toStringHMS() {
        return date("H:i:s", $this->sec); }

    /* Return just the hours-minutes
     * part of the timestamp in the human-readable format.
     */
    public function toStringHM() {
        return date("H:i", $this->sec); }

    /* Produce a string in the following format: DD_HH:MM
     */
    public function toStringDHM() {
        return date("d_H:i", $this->sec); }
        
    /* Return 4-digit year number of the timestamp.
     */
    public function year() { return (int)date("Y", $this->sec); }

    /* Return 2-digit month number of the timestamp.
     */
    public function month() { return (int)date("m", $this->sec); }

    /* Return 2-digit day number of the timestamp.
     */
    public function day() { return (int)date("d", $this->sec); }

    /* Return 1-digit day of week number of the timestamp.
     *
     * Returns:
     *   1 : Monday
     *   2 : Tuesday
     *   ...
     *   7 : Sunday
     */
    public function day_of_week() { return (int)date("N", $this->sec); }

    /* Return 2-digit hour number of the timestamp.
     */
    public function hour() { return (int)date("H", $this->sec); }

    /* Return 2-digit minute number of the timestamp.
     */
    public function minute() { return (int)date("i", $this->sec); }

    /* Return 2-digit second number of the timestamp.
     */
    public function second() { return (int)date("s", $this->sec); }

    /* Convert the tuple into a packed representation of a 64-bit
     * number. These numbers are meant to be stored in a database.
     *
     * NOTE: due to extended range of returned values which is not
     * nativelly supported by PHP the result is returned as a string.
     */
    public function to64() {
        return sprintf("%010d%09d", $this->sec, $this->nsec); }

    /* Produce a packed timestamp from an input value regardless whether
     * it's an object of the current class or it's already a packed
     * representation (64-bit number).
     */
    public static function to64from( $time ) {
        if(is_object($time)) return $time->to64();
        return $time;
    }

    public function to_float() {
        return $this->sec + 1e-9 * $this->nsec; }

    /* Create an object of the current class out of a packed 64-bit
     * numeric representation.
     *
     * NOTE: the "numeric" representation is actually a string.
     */
    public static function from64( $packed ) {
        $sec = 0;
        $nsec = 0;
        $len = strlen( $packed );
        if( $len <= 9 ) {
            if( 1 != sscanf( $packed, "%ud", $nsec ))
                throw new LusiTimeException(
                    __METHOD__, "failed to translate nanoseconds from: '{$packed}'" );
        } else {
            if( 1 != sscanf( substr( $packed, 0, $len - 9), "%ud", $sec ))
                throw new LusiTimeException(
                    __METHOD__, "failed to translate seconds from: '{$packed}'" );

            if( 1 != sscanf( substr( $packed, -9 ), "%ud", $nsec ))
                throw new LusiTimeException(
                    __METHOD__, "failed to translate nanoseconds from: '{$packed}'" );
        }
        return new LusiTime( $sec, $nsec );
    }

    /* Check if the input timestamp falls into the specified interval.
     * The method will return the following values:
     * 
     *    < 0  - the timestamp falls before the begin time of the interval
     *    = 0  - the timestamp is within the interval
     *    > 0  - the timestamp is at or beyon the end time of the interval
     * 
     * The method will throw an exception if either of the timestamp
     * is not valid, or if the end time of the interval is before its begin time.
     * The only exception is the end time, which is allowed to be the null
     * object for open-ended intervals.
     */
    public static function in_interval( $at, $begin, $end ) {

        if( is_null( $at ) || is_null( $begin ))
            throw new LusiTimeException(
                __METHOD__, "timestamps can't be null" );

        $at_obj    = (is_object( $at )    ? $at    : LusiTime::from64( $at ));
        $begin_obj = (is_object( $begin ) ? $begin : LusiTime::from64( $begin ));
        $end_obj = $end;

        if( !is_null( $end ) AND !is_object( $end )) $end_obj = LusiTime::from64( $end );
        if( $at_obj->less( $begin_obj )) return -1;
        if( is_null( $end_obj ) OR $at_obj->less( $end_obj )) return 0;
        return 1;
    }

    /* Compare the current object with the one given in the parameter.
     * Return TRUE if the current object is STRICTLY less than the given one.
     */
    public function less( $rhs ) {
        return ( $this->sec < $rhs->sec ) ||
               (( $this->sec == $rhs->sec ) && ($this->nsec < $rhs->nsec)); }

    public function greaterOrEqual( $rhs ) {
        return $rhs->less( $this ); }

    /* Compare the current object with the one given in the parameter.
     * Return TRUE if the current object is equal to the given one.
     */
    public function equal( $rhs ) {
        return ( $this->sec == $rhs->sec ) && ($this->nsec == $rhs->nsec); }
}
/*
$t = LusiTime::parse( "2010-05-28 14:39:44.123456789" );
print("<br>".$t);
*/
?>
