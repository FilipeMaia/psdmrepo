<?php

/* ATTENTION: This limit is required to deal with huge data structures/collections
 * produced by some PHP functions when dealing with irodsws collections. Consider
 * increasing it further down if the interpreter will stop working and if the Web
 * server's log file /var/log/httpd/error_log will say something like:
 *
 *  ..
 *  Allowed memory size of 16777216 bytes exhausted (tried to allocate 26 bytes)
 *  ..
 */
ini_set("memory_limit","64M");

/*
 * The helper utility class to deal with iRODS Web Servive
 */
class FileMgrIrodsWs {

    private static $base_url = 'https://pswww.slac.stanford.edu/ws-auth/irodsws';
    private static $opts = array(
    	"timeout"      => 10,
    	"httpauthtype" => HTTP_AUTH_BASIC,
    	"httpauth"     => "gapon:newlife2"
    );

    /**
     * Find out a range of runs for a given type.
     *
     * @param String $instrument
     * @param String $experiment
     * @param String $type
     * @return array( <min_run>, <max_run> )
     */
    public static function run_range( $instrument, $experiment, $type ) {

        $runs = null;
        FileMgrIrodsWs::request( $runs, '/runs/'.$instrument.'/'.$experiment.'/'.$type );

        $min = PHP_INT_MAX;
        $max = -1;
        foreach( $runs as $r ) {
            if( $r->run < $min ) $min = $r->run;
            if( $r->run > $max ) $max = $r->run;
        }
        return array( 'min' => $min, 'max' => $max, 'total' => count( $runs ));
    }

    public static function max_run_range( $instrument, $experiment, $types ) {

        $min = PHP_INT_MAX;
        $max = -1;
        $total = 0;

        foreach( $types as $type ) {
            $range = FileMgrIrodsWs::run_range( $instrument, $experiment, $type );
            if( $range['min']   < $min   ) $min   = $range['min'];
            if( $range['max']   > $max   ) $max   = $range['max'];
            if( $range['total'] > $total ) $total = $range['total'];
        }
        
        if( $min > $max || $min <= 0 || $max <= 0 || $total < 0 ) die( "unable to figure out a range of runs" );

        return array( 'min' => $min, 'max' => $max, 'total' => $total );        
    }

    public static function runs( &$result, $instrument, $experiment, $type, $range ) {
        FileMgrIrodsWs::request( $result, '/runs/'.$instrument.'/'.$experiment.'/'.$type.'/'.$range );
    }

    public static function files( &$result, $path='' ) {
        FileMgrIrodsWs::request( $result, '/files'.$path );
    }

    public static function files_only( &$result, $path='' ) {
        $files = null;
        FileMgrIrodsWs::files( $files, $path );
        $result = array();
        foreach( $files as $f ) {
            if( $f->type == 'collection' ) continue;
            array_push( $result, $f );
        }
    }

    public static function collections( &$result, $path='' ) {
        $files = null;
        FileMgrIrodsWs::files( $files, $path );
        $result = array();
        foreach( $files as $f ) {
            if( $f->type != 'collection' ) continue;
            if( $f->name == '/' ) continue;
            array_push( $result, $f );
        }
    }
    
    public static function request( &$result, $service ) {

        $url = FileMgrIrodsWs::$base_url.$service;
        $info = null;

        $response = http_get( $url, FileMgrIrodsWs::$opts, $info );

        if( $info['response_code'] != 200 )
            die( "Web service request faild: {$url}" );

        $response_parsed = http_parse_message( $response );

        // Promote result to a string. This is needed because the body is returned
        // as stdClass which can't be JSON decoded.
        //
        $str = ''.$response_parsed->body;

        $result = json_decode( $str );
    }
}
?>