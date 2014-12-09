<?php

require_once( 'authdb/authdb.inc.php' );
require_once( 'regdb/regdb.inc.php' );

use AuthDB\AuthDB;

use RegDB\RegDB;
use RegDB\RegDBAuth;
use RegDB\RegDBHtml;

/*
 * This script will lay out a form for viewing/editing a user account.
 */

header( 'Content-type: text/html' );
header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

function report_error($msg) {
    print $msg;
    exit;
}

if( isset( $_GET['uid'] )) {
    $uid = trim( $_GET['uid'] );
    if( $uid == '' ) report_error( AuthDB::reporErrorHtml( "user identifier can't be empty" ));
} else {
    report_error( AuthDB::reporErrorHtml( "no valid user identifier" ));
}
$edit = isset( $_GET['edit'] );

/* Proceed with the operation
 */
try {

    RegDB::instance()->begin();

    /* Get all groups the account is member of. The account must exist.
     */
    $account = RegDB::instance()->find_user_account( $uid );
    if( is_null( $account ))
        report_error( RegDBAuth::reporErrorHtml('No such account known to LDAP server: '.$uid ));

    $account_groups = array();
    foreach( $account['groups'] as $g ) $account_groups[$g] = True;

    /* Use this as a cache to avoid showing duplicate entries when
     * displaying groups.
     */
    $displayed_groups = array();
    
    /* Create a container with standard fields
     */
    $con = new RegDBHtml( 0, 0, 1000, 480 );
    $con->label(   0,   0, 'Name:'   )->value( 80,   0, $account['gecos'], false )->hidden_action( 'user:gecos', $account['gecos'] )
        ->label(   0,  25, 'UID:'    )->value( 80,  25, $account['uid'],   false )->hidden_action( 'user:uid',   $account['uid'] )
        ->label(   0,  50, 'E-Mail:' )->value( 80,  50, $account['email'], false )->hidden_action( 'user:email', $account['email'] );

    $col_base = 300;
    $row_base = 0;

    $col = 0;

    /* Experiment-specific groups.
     */
    foreach( RegDB::instance()->instrument_names() as $instr ) {

        /* Skip "locations" of "facilities" which do not have any real POSIX
         * groups neither data associated with them.
         */
        $instrument = RegDB::instance()->find_instrument_by_name( $instr );
        if( $instrument->is_location()) continue;

        /* Skip instruments for which there are no eligible experiments/groups yet.
         */
        $experiment_specific_groups = RegDB::instance()->experiment_specific_groups( $instr );
        if( !count( $experiment_specific_groups )) continue;

        $row = 0;

        foreach( array_keys( $experiment_specific_groups ) as $g ) {

        	$edit_prohibited_flag = !( $edit && RegDBAuth::instance()->canManageLDAPGroup( $g ));

            $displayed_groups[$g] = True;
            if( 1 == preg_match( '/^[a-z]{3}[a-z0-9]+[0-9]{2}$/', $g )) {
                $group_url = "<a href=\"javascript:view_group('".$g."')\">".$g."</a>";
                $row_pos = $row_base + $row * 20;
                $col_pos = $col_base + $col * 120;
                $con->checkbox_input( $col_pos,      $row_pos, "gid:".$g, $g, array_key_exists( $g, $account_groups ), $edit_prohibited_flag /*!$edit*/ )
                    ->label         ( $col_pos + 20, $row_pos, $group_url, false );        	
                $row += 1;
            }
        }
        $col += 1;
    }

    /* Other groups which can't be managed by this application..
     */
    $row = 0;

    foreach( array_keys( $account_groups ) as $g ) {

        if( array_key_exists( $g, $displayed_groups )) continue;

        $group_url = "<a href=\"javascript:view_group('".$g."')\">".$g."</a>";
        $row_pos = $row_base + $row * 20;
        $col_pos = $col_base + $col * 120;
        $con->checkbox_input( $col_pos,      $row_pos, "gid:".$g, $g, true, true )
            ->label         ( $col_pos + 20, $row_pos, $group_url, false );        	
        $row += 1;
    }
        
    echo $con->html();

    RegDB::instance()->commit();

} catch( Exception $e ) { report_error( $e.'<pre>'.print_r( $e->getTrace(), true ).'</pre>' ); }

?>
