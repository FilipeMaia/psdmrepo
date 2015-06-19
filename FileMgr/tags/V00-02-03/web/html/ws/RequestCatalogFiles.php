<?php

require_once( 'authdb/authdb.inc.php' );
require_once( 'filemgr/filemgr.inc.php' );
require_once( 'regdb/regdb.inc.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

use FileMgr\FileMgrIrodsWs;
use FileMgr\FileMgrException;

use RegDB\RegDBHtml;

// Make sure we're using the rigth tomezone. Otherwise PHP will complain
// into Web server's log files.
//
date_default_timezone_set('America/Los_Angeles');

/*
 * This script will process requests for various information stored in the database.
 * The result will be returned an embedable HTML element (<div>).
 */
if( !isset( $_GET['path'] )) die( "no valid iRODS catalog path" );
$path = trim( $_GET['path'] );

function pre( $str, $width=null ) {
    if( is_null( $width )) return '<pre>'.$str.'</pre>';
    return '<pre>'.sprintf( "%{$width}s", $str ).'</pre>';
}

/*
 * Analyze and process the request
 */
try {

    AuthDB::instance()->begin();

    $files = null;    
    FileMgrIrodsWs::files_only( $files, $path );

    //print_r( $files );

    header( 'Content-type: text/html' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past
    
    /* Count the number of files as we need to determine a number of rows
     * in the RegDBHtml below.
     */
    $num_rows = 20 * count( $files );

    $con = new RegDBHtml( 0, 0, 850, $num_rows );

    $row = 0;
    foreach( $files as $file ) {

        $name     = pre( $file->name );
        $owner    = pre( $file->owner );
        $size     = pre( number_format( $file->size ), 17 );    // less than 10 TB
        $created  = pre( date( "Y-m-d H:i:s", $file->ctime ));
        $resource = pre( $file->resource );
        $location = pre( '<a href="javascript:display_path('."'".$file->path."'".')">path</a>' );

        $color = null;
        if( $file->resource == 'hpss-resc' ) $color = '#c0c0c0';


        $con->value(   5, $row, $name,     $color );
        $con->value( 260, $row, $owner,    $color );
        $con->value( 340, $row, $size,     $color );
        $con->value( 500, $row, $created,  $color );
        $con->value( 670, $row, $resource, $color );
        $con->value( 790, $row, $location, $color );

        $row += 20;
    }
    print $con->html();

    AuthDB::instance()->commit();

} catch (AuthDBException  $e ) { print $e->toHtml(); }
  catch (FileMgrException $e ) { print $e->toHtml(); }

?>
