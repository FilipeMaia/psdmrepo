<?php

/*
 * The class representing time in Web applications. It has
 * two data members representing the number of seconds since UNIX
 * epoch (GMT timezone) and the number of nanoseconds. The number
 * of nanoseconds is in the range of 0..999999999.
 */
class LogBookTime {

    public $sec;
    public $nsec;

    public static function now() {
        return new LogBookTime( mktime());
    }
    /*
     * Parse an input string into an object of the class.
     *
     * Input formats:
     *
     *   2009-05-19 17:59:49-07
     *   2009-05-19 17:59:49-07:00
     *   2009-05-19 17:59:49-0700
     *   2009-05-19 12:18:49.123456789-0700
     *
     * The last example illustrates how to specify nanoseconds. The number
     * of nanoseconds (if not empty) must be in the range of 0..999999999.
     *
     * The method will return an object of the class or it will
     * return FALSE in case of any error.
     */
    public static function parse($str) {

        $expr = '/(\d{4})-(\d{1,2})-(\d{1,2}) (\d{1,2}):(\d{1,2}):(\d{1,2})(\.(\d{1,9}))?(([-+])(\d{2}):?(\d{2})?)/';
        if( !preg_match( $expr, $str, $matches2 ))
            return FALSE;

        //print_r($matches2);

        $sign = $matches2[10];
        $tzone_hours = $matches2[11];
        $tzone_minutes = isset( $matches2[12] ) ? $matches2[12] : 0;
        $shift = ($sign=='-' ? +1 : -1)*(3600*$tzone_hours + 60*$tzone_minutes);

        //print( "TimeZone shift to apply [seconds]: ".$shift."\n" );

        $local_time = strtotime( $str );
        if( !$local_time )
            return FALSE;

        $gmt_time = ($local_time+$shift);

        //print( "Input time             : ".$str."\n" );
        //print( "Base time              : ".$local_time."\n" );
        //print( "GMT  time              : ".$gmt_time."\n" );
        //print( "Current UNIX timestamp : ".mktime()."\n" );

        $nsec = isset( $matches2[8] ) ? $matches2[8] : 0;

        return new LogBookTime($gmt_time,$nsec);
    }

    public function __construct($sec, $nsec=0) {
        if( $nsec < 0 or $nsec > 999999999)
            die( "the number of nanoseconds isn't in allowed range" );
        $this->sec = $sec;
        $this->nsec = $nsec;
    }

    public function __toString() {
        return sprintf("%010d.%09d", $this->sec, $this->nsec );
    }

    /* Convert the tuple into a packed representation of a 64-bit
     * number. These numbers are meant to be stored in a database.
     *
     * NOTE: due to extended range of returned values which is not
     * nativelly supported by PHP the result is returned as a string.
     */
    public function to64() {
        return sprintf("%010d%09d", $this->sec, $this->nsec);
        /*
        return
            gmp_strval(
                gmp_add(
                    gmp_mul(
                        "1000000000",
                        $this->sec ),
                    gmp_init( $this->nsec )));
         */
    }
}
/*
echo "here follows a simple unit test for the class.\n";

$str = "2009-05-19 17:59:49.123-0700";

$lt = LogBookTime::parse( $str) ;

print( "  Input time:           ".$str."\n" );
print( "  LogBookTime::parse(): ".$lt->__toString()."\n" );
print( "  converted to 64-bit:  ".$lt->to64()."\n" );
*/
?>
