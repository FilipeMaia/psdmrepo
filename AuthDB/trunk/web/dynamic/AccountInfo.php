<?php

require_once('AuthDB/AuthDB.inc.php');
require_once('RegDB/RegDB.inc.php');

/*
 * This script will lay out a form for viewing/editing a user account.
 */
if( isset( $_GET['uid'] )) {
    $uid = trim( $_GET['uid'] );
    if( $uid == '' ) {
        print( AuthDB::reporErrorHtml( "user identifier can't be empty" ));
        exit;
    }
} else {
    print( AuthDB::reporErrorHtml( "no valid user identifier" ));
    exit;
}
$edit = isset( $_GET['edit'] );

if( $edit ) {
    if( !AuthDB::instance()->canEdit()) {
        print( AuthDB::reporErrorHtml(
            'You are not authorized to manage the contents of the LDAP server'));
        exit;
    }
}

/* Proceed with the operation
 */
try {

    $regdb = new RegDB();
    $regdb->begin();

    /* Get all groups the account is member of. The account must exist.
     */
    $account = $regdb->find_user_account( $uid );
    if( is_null( $account )) {
        print( RegDBAuth::reporErrorHtml(
            'No such account known to LDAP server: '.$uid ));
        exit;
    }
    $account_groups = array();
    foreach( $account['groups'] as $g ) {
    	$account_groups[$g] = True;
    }

    /* Use this as a cache to avoid showing duplicate entries when
     * displaying groups.
     */
    $displayed_groups = array();

    header( 'Content-type: text/html' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past
    
    /* Create a container with standard fields
     */
    $con = new RegDBHtml( 0, 0, 800, 480 );
    $con->label(   0,   0, 'Name:'   )->value( 80,   0, $account['gecos'], false )->hidden_action( 'user:gecos', $account['gecos'] )
        ->label(   0,  25, 'UID:'    )->value( 80,  25, $account['uid'],   false )->hidden_action( 'user:uid',   $account['uid'] )
        ->label(   0,  50, 'E-Mail:' )->value( 80,  50, $account['email'], false )->hidden_action( 'user:email', $account['email'] );

    $col_base = 300;
    $row_base = 0;

    $col = 0;

    /* Experiment-specific groups.
     */
    foreach( $regdb->instrument_names() as $instr ) {

        /* Skip "locations" of "facilities" which do not have any real POSIX
         * groups neither data associated with them.
         */
        $instrument = $regdb->find_instrument_by_name( $instr );
        if( $instrument->is_location()) continue;

        /* Skip instruments for which there are no eligible experiments/groups yet.
         */
        $experiment_specific_groups = $regdb->experiment_specific_groups( $instr );
        if( !count( $experiment_specific_groups )) continue;

        $row = 0;

        foreach( array_keys( $experiment_specific_groups ) as $g ) {

            $displayed_groups[$g] = True;
            if( 1 == preg_match( '/^[a-z]{3}[a-z0-9]+[0-9]{2}$/', $g )) {
                $group_url = "<a href=\"javascript:view_group('".$g."')\">".$g."</a>";
                $row_pos = $row_base + $row * 20;
                $col_pos = $col_base + $col * 120;
                $con->checkbox_input( $col_pos,      $row_pos, "gid:".$g, $g, array_key_exists( $g, $account_groups ), !$edit)
                    ->label         ( $col_pos + 20, $row_pos, $group_url, false );        	
                $row += 1;
            }
        }
        $col += 1;
    }

    /* Special groups.
     */
    $row = 0;
    foreach( array_keys( $regdb->preffered_groups()) as $g ) {
   	    $displayed_groups[$g] = True;
        $group_url = "<a href=\"javascript:view_group('".$g."')\">".$g."</a>";
        $row_pos = $row_base + $row * 20;
        $col_pos = $col_base + $col * 120;
        $con->checkbox_input( $col_pos,      $row_pos, "gid:".$g, $g, array_key_exists( $g, $account_groups ), !$edit)
            ->label         ( $col_pos + 20, $row_pos, $group_url, false );        	
        $row += 1;
    }

    /* Append remaining groups by the end of the previous collumn
     */
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

    $regdb->commit();

} catch( AuthDBException $e ) {
    print $e->toHtml();
} catch( RegDBException $e ) {
    print $e->toHtml();}
?>
