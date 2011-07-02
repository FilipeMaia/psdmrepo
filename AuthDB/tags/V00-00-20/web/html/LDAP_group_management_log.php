<?php
require_once( 'authdb/authdb.inc.php' );
require_once( 'regdb/regdb.inc.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;
use AuthDB\Logger;

use RegDB\RegDB;
use RegDB\RegDBException;

/* The script for reporting the LDAP group management operations
*/

$as_text = isset( $_GET['as_text']);

try {
	$authdb = AuthDB::instance();
	$authdb->begin();

	if( !$authdb->hasPrivilege( $authdb->authName(), null, 'Logger', 'read' ))
		die ( <<<HERE
<br><br>
<center>
  <span style="color: red; font-size: 150%; font-weight: bold; font-family: Times, sans-serif;">
    A c c e s s &nbsp; E r r o r
  </span>
</center>
<div style="margin: 10px 10% 10px 10%; padding: 10px; font-size: 125%; font-family: Times, sans-serif; border-top: 1px solid #b0b0b0;">
  We're sorry! Your SLAC UNIX account <b>{$authdb->authName()}</b> has no proper permissions to view
  the contents of this page. If you think you should then  please contact us by sending an e-mail request
  to <b>pcds-help</b> (at SLAC).
</div>
HERE
		);
?>

<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
<head>

<title>Report LDAP Management operations for POSIX Groups</title>
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8"> 
<?php

	$logger = Logger::instance();
	$logger->begin();
	$entries = $logger->get_group_management();

	if( $as_text ) {

		print "<pre>\n";
		$line = '';
		foreach( $entries as $entry ) {
			print
				$entry['event_time']->toStringShort()."  ".
				$entry['requestor']."@".$entry['requestor_host']."  ".
				$entry['class'].": ".$entry['operation']." '".
				$entry['user_account']."' ".$entry['group_type']." '".$entry['group_name']."'".
				"\n";
		}
		print "</pre>\n";

	} else {

?>

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
  padding: 2px 8px 2px 8px;
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

  <h2>About</h2>
  <div style="padding-left:20px;">
    <p>The information found on this page represents a report for recorded
       group management operations performed on user groups (POSIX or NETGROUPS) in the PCDS LDAP
       server. Note that a scope of the report is limited to operations
       requested via PCDS Web tools.</p>
    <p>The report can be also downloaded in the plain text format from <a href="?as_text" target="_blank">here</a>.</p>
  </div>
  <h2>Group Management Operations</h2>
  <div style="padding-left:20px;">

<?php

	print <<<HERE
    <table><tbody>
      <tr>
        <td class="table_hdr">Time</td>
        <td class="table_hdr">Requestor</td>
        <td class="table_hdr">Host</td>
        <td class="table_hdr">Class</td>
        <td class="table_hdr">Operation</td>
        <td class="table_hdr">Account</td>
        <td class="table_hdr">Group Type</td>
        <td class="table_hdr">Group Name</td>
      </tr>
HERE;

	foreach( $entries as $entry ) {
	
		$time           = $entry['event_time']->toStringShort();
		$requestor      = $entry['requestor'];
		$requestor_host = $entry['requestor_host'];
		$class          = $entry['class'];
		$operation      = $entry['operation'];
		$user           = $entry['user_account'];
		$group          = $entry['group_name'];
		$group_type     = $entry['group_type'];

		print <<<HERE
      <tr>
        <td class="table_cell table_cell_left">{$time}</td>
	    <td class="table_cell"><a href="../authdb/?action=view_account&uid={$requestor}">{$requestor}</a></td>
        <td class="table_cell">{$requestor_host}</td>
        <td class="table_cell">{$class}</td>
        <td class="table_cell">{$operation}</td>
        <td class="table_cell"><a href="../authdb/?action=view_account&uid={$user}">{$user}</a></td>
        <td class="table_cell">{$group_type}</td>
        <td class="table_cell table_cell_right"><a href="../authdb/?action=view_group&gid={$group}">{$group}</a></td>
      </tr>
HERE;
	}
	print <<<HERE
    </tbody><table>
  </div>
</div>
HERE;
	}
	$authdb->commit();
	
} catch( AuthDBException  $e ) { print $e->toHtml(); }
  catch( RegDBException   $e ) { print $e->toHtml(); }

?>

</body>
</html>