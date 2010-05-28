<?php

require_once('RegDB/RegDB.inc.php');

/*
 * This script will lay out a form for viewing/editing a user account.
 */
if( isset( $_GET['uid'] )) {
    $uid = trim( $_GET['uid'] );
    if( $uid == '' ) {
        print( RegDBAuth::reporErrorHtml( "user identifier can't be empty" ));
        exit;
    }
} else {
    print( RegDBAuth::reporErrorHtml( "no valid user identifier" ));
    exit;
}
$edit = isset( $_GET['edit'] );

if( $edit ) {
    if( !RegDBAuth::instance()->canEdit()) {
        print( RegDBAuth::reporErrorHtml(
            'You are not authorized to manage the contents of the LDAP server'));
        exit;
    }
}

/* This list of preferred groups will be extended with
 * experiment specific groups. Note, that we define this
 * arrays as a dictionary to simplify searches in it.
 */
$preffered_groups = array(
    'lab-admin'      => True,
    'lab-superusers' => True,
    'lab-users'      => True,
    'ps-amo'         => True,
    'ps-data'        => True,
    'ps-mgt'         => True,
    'xr'             => True,
    'xs'             => True,
    'xu'             => True
);

/* Proceed with the operation
 */
try {

    $regdb = new RegDB();
    $regdb->begin();

    // Find experiment specific groups and add them to the list
    //
    $experiments = $regdb->experiments();
    $groups = $regdb->posix_groups();
    foreach( $experiments as $e ) {
    	$group = $e->name;
    	if( key_exists( $group, $groups )) $preffered_groups[$group] = True;
    }

    // The account must exist.
    //
    $account = $regdb->find_user_account( $uid );
    if( is_null( $account )) {
        print( RegDBAuth::reporErrorHtml(
            'NO such account known to LDAP server: '.$uid ));
        exit;
    }
    
    header( 'Content-type: text/html' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past
    
    /* Create a container with standard fields
     */
    $con = new RegDBHtml( 0, 0, 800, 400 );
    $con->label(   0,   0, 'Name:'   )->label( 80,   0, $account['gecos'], false )
        ->label(   0,  25, 'UID:'    )->label( 80,  25, $account['uid'],   false )
        ->label(   0,  50, 'E-Mail:' )->label( 80,  50, $account['email'], false );

    $row_base = 80;
    $col_base = 80;
    $row = 0;
    $col = 0;
    $con->label(   0,  $row_base, 'Groups:' );
    //foreach( $account['groups'] as $g ) {
    foreach( array_keys( $preffered_groups ) as $g ) {
        $group_url = "<a href=\"javascript:view_group('".$g."')\">".$g."</a>";
    	$row_pos = $row_base + $row * 20;
    	$col_pos = $col_base + $col * 150;
        $con->label( $col_pos, $row_pos, $group_url, false );
        $col += 1;
        if( $col >= 2 ) {
        	$col = 0;
            $row += 1;
        }
    }
    echo $con->html();

    $regdb->commit();

} catch( RegDBException $e ) {
    print $e->toHtml();
}
?>
