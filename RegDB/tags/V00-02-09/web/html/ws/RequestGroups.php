<?php

require_once( 'regdb/regdb.inc.php' );

use RegDB\RegDB;
use RegDB\RegDBAuth;
use RegDB\RegDBException;

if( !RegDBAuth::instance()->canRead()) {
    print( RegDBAuth::reporErrorHtml(
        'You are not authorized to read the group info'));
    exit;
}

// In the 'grid' mode the service will return an embeddable HTML document with
// a full set of all known groups.
// 
$grid = null;
if( isset( $_GET['grid'] )) {
	$grid = trim( $_GET['grid'] );
	if( !( $grid == 'vertical' || $grid == 'horizontal' ))
		die( "illegal value of the 'grid'. Allowed values are 'vertical' or 'horizontal'" );
}

$NUM_COLUMNS = 12;

function groupUrl( $group ) {
	return "<a href=\"javascript:view_group('".$group."')\">".$group.'</a>';
}

function group2json( $group ) {
    return json_encode(
        array ( "group" => groupUrl( $group ))
    );
}

/*
 * Return JSON objects with a list of groups.
 */
try {
    $all_groups = false;    // LCLS specific groups only
    RegDB::instance()->begin();
    $groups = RegDB::instance()->posix_groups($all_groups);

    // Choose the desired presentation
    //
    if( !is_null( $grid )) {

    	header( 'Content-type: text/html' );
    	header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    	header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

        echo "<table><thead>\n";

        if( $grid == 'vertical' ) {

        $num_groups = count( $groups );
        $num_rows   = floor( $num_groups / $NUM_COLUMNS ) + ( $num_groups % $NUM_COLUMNS == 0 ? 0 : 1 );
        $rows = array();
        for( $i = 0; $i < $num_rows; $i++ ) array_push( $rows, "  <tr>\n    " );
        $r = 0;
        foreach( $groups as $g ) {
            $rows[$r] .= '<td><div style="padding:2px; padding-right:10px;">'.groupUrl( $g ).'</div></td>';
            if( ++$r >= $num_rows ) $r = 0;
        }
        for( $i = 0; $i < $num_rows; $i++ ) echo $rows[$i]."\n  </tr>\n";
	   		
    	} else if( $grid == 'horizontal' ) {

            $num_groups       = count( $groups );
            $num_full_rows    = floor( $num_groups / $NUM_COLUMNS );
            $cols_in_last_row = $num_groups % $NUM_COLUMNS;

            for( $r = 0; $r < $num_full_rows; $r++ ) {
                $base = $r*$NUM_COLUMNS;
                echo "  <tr>\n    ";
                for( $c = 0; $c < $NUM_COLUMNS; $c++ ) {
                    echo '<td><div style="padding:2px;">'.groupUrl( $groups[$base+$c] ).'</div></td>';
                }
                echo "\n  </tr>\n";
            }
            if( $cols_in_last_row > 0 ) {
                $base = $num_groups - $cols_in_last_row;
                echo "  <tr>\n    ";
                for( $c = 0; $c < $cols_in_last_row; $c++ ) {
                    echo '<td><div style="padding:2px;">'.groupUrl( $groups[$base+$c] ).'</div></td>';
                }
                echo "\n  </tr>\n";
            }
    	}
    	echo "</thead></table>\n";

    } else {
	
    	header( 'Content-type: application/json' );
    	header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    	header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

    	print <<< HERE
{
  "ResultSet": {
    "Result": [
HERE;
    	$first = true;
    	foreach( $groups as $g ) {
        	if( $first ) {
            	$first = false;
            	echo "\n".group2json( $g );
        	} else {
            	echo ",\n".group2json( $g );
        	}
    	}
    	print <<< HERE
 ] } }
HERE;
    }
    RegDB::instance()->commit();

} catch( RegDBException $e ) { print $e->toHtml(); }

?>
