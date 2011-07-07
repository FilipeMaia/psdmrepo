<?php

require_once( 'authdb/authdb.inc.php' );
require_once( 'logbook/logbook.inc.php' );
require_once( 'regdb/regdb.inc.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

use LogBook\LogBook;
use LogBook\LogBookException;

use RegDB\RegDB;
use RegDB\RegDBException;

/** Harvest the optional parameters first.
 */
if( isset( $_GET['instr_name'] )) {
	$instr_name = strtoupper( trim( $_GET['instr_name'] ));
	if( $instr_name == '' ) die( "<b>error:</b> instrument name parameter if present can't have an empty value" );
}
$fix = isset( $_GET['fix'] );

try {
	$authdb = AuthDB::instance();
	$authdb->begin();

	if( $fix && !$authdb->canEdit()) die( "<b>error:</b> your account doesn't posseses sufficient privileges to perform the operaton" );

	$logbook = new LogBook();
	$logbook->begin();

	$instrument_names = array();
	if( isset( $instr_name )) {
		if( is_null( $logbook->regdb()->find_instrument_by_name( $instr_name ))) die( "<b>error:</b> the specified instrument isn't known" );
		array_push( $instrument_names, $instr_name  );
	} else {

		// Get all known instruments. Skip  pseudo-instruments.
		//
		foreach( $logbook->regdb()->instrument_names() as $name ) {
			$instrument = $logbook->regdb()->find_instrument_by_name($name);
			if( $instrument->is_location()) continue;
			array_push( $instrument_names, $name  );
		}
	}
	$all_instrument_names_subpattern = '';
	foreach( $logbook->regdb()->instrument_names() as $name ) {
		$instrument = $logbook->regdb()->find_instrument_by_name($name);
		if( $instrument->is_location()) continue;
		if( $all_instrument_names_subpattern != '') $all_instrument_names_subpattern .= '|';
		$all_instrument_names_subpattern .= strtolower( $name );
	}
	
	$all_accounts = $logbook->regdb()->user_accounts();

	$extra_operations = <<<HERE
  <h2>Extra Operations</h2>
  <div style="padding-left:20px;">
	<ul>
      <li><a href="LDAPManagers.php">check</a> or <a href="LDAPManagers.php?fix">fix</a> authorizations status of all instruments</li>
HERE;
	foreach( $instrument_names as $name ) {
		$extra_operations .= <<<HERE
      <li><a href="LDAPManagers.php?instr_name={$name}">check</a> or <a href="LDAPManagers.php?instr_name={$name}&fix">fix</a> authorizations status for instrument {$name}</li>
HERE;
	}
	$extra_operations .= <<<HERE
    </ul>
  </div>
HERE;

?>

<!DOCTYPE html"> 
<html>
<head>

<title>Authorize PIs to manage LDAP Groups of their Experiments</title>
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

  <h2>About</h2>
  <div style="padding-left:20px;">
    <p>This script will check if experiment PIs are authorized as 'Admin's of POSIX groups
       of their experiments, and if the optional parameter <b>fix</b> is present then assign
       the role to the PIs. Specifically, for each experiment, the tool will:</p>
    <ol>
      <li>find a POSIX group associated with the experiment</li>
      <li>get a PI account associated with the experiment (normally this is a shared account assigned to the real PI's name)</li>
      <li>if the account has the same name as the experiment then look for another UNIX account whose owner matches the owner's name of the shared account</li>
      <li>if one (and only one!) such account is found then check if the group is already managed by that account</li>
      <li>(optionally) if not then authorize the account as 'Admin' of the experiment's group</li>
      <li>(optionally) report which changes which had to be made at step #5</li></ol>
    <p>The script has two optional parameters:</p>
    <ul>
      <li><b>instr_name='INSTRUMENT'</b> - limit a scope of the operation to the specified instrument</li>
      <li><b>fix</b> - proceed with the actual fix and authorize the groups. By default</li></ul>
  </div>
  <?php echo $extra_operations; ?>
  <h2>Authorizations</h2>
  <div style="padding-left:20px;">

<?php

	$experiments_by_names = array();
	foreach( $logbook->experiments() as $e ) $experiments_by_names[$e->name()] = $e;

	$application = 'LDAP';
	$role = 'Admin';
	$header =
		'<tr>'.
		'<td class="table_hdr">Instrument</td>'.
		'<td class="table_hdr">Id</td>'.
		'<td class="table_hdr">Experiment</td>'.
		'<td class="table_hdr">Group</td>'.
		'<td class="table_hdr">PI (regdb)</td>'.
		'<td class="table_hdr">PI (matched)</td>'.
		'<td class="table_hdr">LDAP Role</td>'.
		'<td class="table_hdr">Is Authorized</td>'.
		'<td class="table_hdr">Current authorizations</td>'.
	($fix ? '<td class="table_hdr">Action</td>' : '').
	'</tr>';
	print '<table><tbody>';
	foreach( $instrument_names as $instr_name ) {

		print $header;

		$experiments = $logbook->experiments_for_instrument( $instr_name );
		foreach( $experiments as $experiment ) {

			// Get the real UNIX account of the PI if the one has such account.
			// A problem is that at a time when we're reegistering new experiments many PIs
			// may still not have personal UNIX accounts at SLAC, so instead we're using
			// shared accounts of their experiments. The simple algorithm below will try
			// to guess the real PIs by obtaining the owner's name of the shared account
			// and using that information to search for some other account. And if that would
			// result in EXACTLY two accounts (the first one is the shared account) then we'll
			// asssume the second one as the real account.
			// 
			$real_pi_uid = $experiment->leader_account();
			$alledged_pi_uids_found = array();
			if( $real_pi_uid == $experiment->name()) {
				$account = $logbook->regdb()->find_user_account( $real_pi_uid );

				// Extract first and last names
				//
				if( preg_match('/^([^\s]+)\s+([^\s]+)$/',          $account['gecos'], $result) ||
				    preg_match('/^([^\s]+)\s+[^\s]*\s+([^\s]+)$/', $account['gecos'], $result)) {

				    $first_name = $result[1];
					$last_name  = $result[2];

					//print $account['gecos'].' : '.$first_name.' '.$last_name.' [<span style="color:green;">OK</span>]<br>';
					foreach( $all_accounts as $a ) {

						if( preg_match( '/^('.$all_instrument_names_subpattern.')\d{5}$/', $a['uid'] )) continue;

						// Look for a combination of the first and last names (both must be
						// present accross all known accounts. (Obviously) ignore the shared
						// account (the one we're starting with).
						//
						if( preg_match('/^'.$first_name.'\s+'.$last_name.'$/',           $a['gecos']) ||
						    preg_match('/^'.$first_name.'\s+[^\s]*\s+'.$last_name.'$/',  $a['gecos']) ||
						    preg_match('/^'.$last_name.',\s+'.$first_name.'$/',          $a['gecos']) ||
						    preg_match('/^'.$last_name.',\s+'.$first_name.'\s+[^\s]*$/', $a['gecos'])) {
						    //print '&nbsp;&nbsp;&nbsp;&nbsp;'.$a['uid'].'<br>';
						    array_push( $alledged_pi_uids_found, $a['uid'] );
						}
					}
				} else {
					//print $account['gecos'].' : [<span style="color:red;">FAILED</span>]<br>';
				}
			}
			$alledged_pi_uid = '';
			if( count( $alledged_pi_uids_found ) == 1 ) $alledged_pi_uid = $alledged_pi_uids_found[0];

			$has_role =
				$alledged_pi_uid == '' ?
				$authdb->hasRole( $real_pi_uid,     $experiment->id(), $application, $role ) || $authdb->hasRole( $real_pi_uid,     null, $application, $role ) :
				$authdb->hasRole( $alledged_pi_uid, $experiment->id(), $application, $role ) || $authdb->hasRole( $alledged_pi_uid, null, $application, $role );

			print
				'<tr>'.
				'<td class="table_cell table_cell_left">'.$instr_name.'</td>'.
				'<td class="table_cell">'.$experiment->id().'</td>'.
			    '<td class="table_cell"><a href="../portal/?exper_id='.$experiment->id().'">'.$experiment->name().'</a></td>'.
				'<td class="table_cell"><a href="../authdb/?action=view_group&gid='.$experiment->POSIX_gid().'">'.$experiment->POSIX_gid().'</a></td>'.
				'<td class="table_cell"><a href="../authdb/?action=view_account&uid='.$real_pi_uid.'">'.$real_pi_uid.'</a></td>';
			if( $alledged_pi_uid == '' )
				print
				'<td class="table_cell"></td>';
			else
				print
				'<td class="table_cell"><a href="../authdb/?action=view_account&uid='.$alledged_pi_uid.'">'.$alledged_pi_uid.'</a></td>';
			print
				'<td class="table_cell">'.$role.'</td>';

			// TODO: Make sure the above found (if any) 'aledged' PI is authorized
			//       to manage their group.
			//
			if( $fix ) {
				if( !$has_role  ) {
					if( $real_pi_uid != $experiment->name()) {
						//$authdb->createRolePlayer( $application, $role, $experiment->id(), $group4auth );
						print
							'<td class="table_cell"><span style="color:red; font-weight:bold;">No</span></td>'.
							'<td class="table_cell"></td>'.
							'<td class="table_cell table_cell_right"><span style="color:red; font-weight:bold;">fix is not implemented yet</span></td>';
					} else {
						print
							'<td class="table_cell"><span style="color:red; font-weight:bold;">No</span></td>'.
							'<td class="table_cell"></td>'.
							'<td class="table_cell table_cell_right"></td>';
						}
				} else {
					print
						'<td class="table_cell">Yes</td>'.
						'<td class="table_cell"></td>'.
						'<td class="table_cell table_cell_right"></td>';
				}
			} else {
				print
					'<td class="table_cell">'.($has_role ? 'Yes' : '<span style="color:red; font-weight:bold;">No</span>').'</td>'.
					'<td class="table_cell table_cell_right"></td>';
			}
			print
				'</tr>';
		}
	}
	print '</tbody><table>';
	
	$logbook->commit();
	$authdb->commit();
	
} catch( AuthDBException  $e ) { print $e->toHtml(); }
  catch( LogBookException $e ) { print $e->toHtml(); }
  catch( RegDBException   $e ) { print $e->toHtml(); }
  
?>

  </div>
</div>

</body>
</html>