<?php

# Needed to supress complains in the server's log files
date_default_timezone_set('America/Los_Angeles') ;

require_once( 'lusitime/lusitime.inc.php' );

use LusiTime\LusiTime;
use LusiTime\LusiTimeException;

function report_error($msg) {
    print $msg;
    exit;
}
try {
    $begin = LusiTime::parse('2009-10-28 21:10:21');
    $end   = LusiTime::parse('2009-10-28 21:16:05');
    print <<<HERE
<br>{$begin->toStringShort()} : {$begin->to64()}
<br>{$end->toStringShort()} : {$end->to64()}
HERE;

} catch( LusiTimeException   $e ) { report_error( $e.'<pre>'.print_r( $e->getTrace(), true ).'</pre>' ); }

?>
