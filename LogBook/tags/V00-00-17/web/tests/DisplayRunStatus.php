<?php

require_once('LogBook/LogBook.inc.php');
require_once('RegDB/RegDB.inc.php');

try {
    header( 'Content-type: text/html' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

    $con = new RegDBHtml( 0, 0, 700, 250 );
    echo $con
        ->label( 0,  0, 'Run #' )->value( 50,  0, '80195' )->label( 200, 0, 'Type' )->value( 250, 0, 'CALIB' )
        ->label( 0, 25, 'Start' )->value( 50, 25, '2009-06-29 23:45:05' )
        ->label( 0, 50, 'Stop'  )->value( 50, 50, '2009-06-30 00:15:32' )
        ->html();

} catch( RegDBException $e ) {
    print $e->toHtml();
}
?>