<?php

require_once( 'authdb/authdb.inc.php' );
require_once( 'logbook/logbook.inc.php' );
require_once( 'regdb/regdb.inc.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

use RegDB\RegDB;
use RegDB\RegDBException;

try {
	$authdb = AuthDB::instance();
	$authdb->begin();

	$regdb = new RegDB();
	$regdb->begin();

?>

<!DOCTYPE html"> 
<html>
<head>

<title>Experiments members not registered in LDAP groups ps-users or lab-users</title>
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8"> 

<style type="text/css">

td.table_hdr {
  background-color:#d0d0d0;
  padding: 2px 8px 2px 8px;
  border: solid 1px #000000;
  border-top: none;
  border-left: none;
  font-family: Arial, sans-serif;
  font-weight: bold;
  font-size: 75%;
}
td.table_cell {
  border:solid 1px #d0d0d0;
  border-top: none;
  border-left: none;
  padding: 4px 8px 4px 8px;
  font-family: Arial, sans-serif;
  font-size: 75%;
}
td.table_cell_left {
  font-weight: bold;
}
td.table_cell_right {
  border-right: none;
}
td.table_cell_bottom {
  border-bottom: none;
}
td.table_cell_within_group {
  border-bottom: none;
}
</style>
</head>

<body>

<div style="padding-left:20px; padding-right:20px;">

  <h2>Members of POSIX group 'xu' not registered in groups 'ps-users' or 'lab-users'</h2>
  <div style="padding-left:20px;">
   <table><tbody>
     <tr>
       <td class="table_hdr">Account</td>
       <td class="table_hdr">User</td>
     </tr>

<?php

	$experiments_by_names = array();
	foreach( $regdb->experiments() as $experiment ) {
		if( $experiment->is_facility()) continue;
		$experiments_by_names[$experiment->name()] = $experiment;
	}

	$accounts_by_names = array();
	foreach( $regdb->posix_group_members( 'xu' ) as $account ) {

		$uid   = $account['uid'];
		$gecos = $account['gecos'];

		// Skip processed accounts
		//
		if( array_key_exists( $uid, $accounts_by_names )) continue;
		$accounts_by_names[$uid] = True;

		// Skip shared accounts
		//
		if( array_key_exists( $uid, $experiments_by_names )) continue;

		if( ! ( $regdb->is_member_of_posix_group( 'ps-users',  $uid ) ||
			$regdb->is_member_of_posix_group( 'lab-users', $uid ))) {
			print <<<HERE
      <tr>
        <td class="table_cell table_cell_left"><a href="../authdb/?action=view_account&uid={$uid}">{$uid}</a></td>
        <td class="table_cell table_cell_right">{$gecos}</td>
      </tr>

HERE;
		}
	}
	$regdb->commit();
	$authdb->commit();
	
} catch( AuthDBException  $e ) { print $e->toHtml(); }
  catch( LogBookException $e ) { print $e->toHtml(); }
  catch( RegDBException   $e ) { print $e->toHtml(); }
  
?>

    </tbody><table>
  </div>
</div>

</body>
</html>
