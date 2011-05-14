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

	$extra_operations = <<<HERE
  <h2>Extra Operations</h2>
  <div style="padding-left:20px;">
	<ul>
      <li><a href="AuthInstrGroups.php">check</a> or <a href="AuthInstrGroups.php?fix">fix</a> authorizations status of all instruments</li>
HERE;
	foreach( $instrument_names as $name ) {
		$extra_operations .= <<<HERE
      <li><a href="AuthInstrGroups.php?instr_name={$name}">check</a> or <a href="AuthInstrGroups.php?instr_name={$name}&fix">fix</a> authorizations status for instrument {$name}</li>
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

<title>Authorize Instrument Groups for e-Logs of Experiments</title>
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8"> 

<style type="text/css">

td.table_hdr {
  background-color:#d0d0d0;
  padding: 4px 8px 4px 8px;
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
    <p>This script will check if instrument-specific POSIX groups are authorized as e-log 'Writer's of
       the corresponding experiments, and if the optional parameter <b>fix</b> is present then assign
       the role to the groups. Specifically, for each instrument 'XYZ', the tool will:</p>
    <ol>
      <li>check if there is such POSIX group as 'ps-xyz'</li>
      <li>check if the group is registered as e-log 'Writer' for all experiments of instrument 'XYZ'</li>
      <li>(optionally) if not then the group will be authorized for those experiments</li>
      <li>(optionally) report which changes which had to be made at step #3</li></ol>
    <p>The script has two optional parameters:</p>
    <ul>
      <li><b>instr_name='INSTRUMENT'</b> - limit a scope of the operation to the specified instrument</li>
      <li><b>fix</b> - proceed with the actual fix and authorize the groups. By default</li></ul>
  </div>
  <?php echo $extra_operations; ?>
  <h2>Authorizations</h2>
  <div style="padding-left:20px;">

<?php

	$application = 'LogBook';
	$role = 'Writer';
	$header =
		'<tr>'.
		'<td class="table_hdr">Instrument</td>'.
		'<td class="table_hdr">Id</td>'.
		'<td class="table_hdr">Experiment</td>'.
		'<td class="table_hdr">Group</td>'.
		'<td class="table_hdr">Instrument Group</td>'.
		'<td class="table_hdr">E-Log Role</td>'.
		'<td class="table_hdr">Is Authorized</td>'.
		($fix ? '<td class="table_hdr">Action</td>' : '').
	'</tr>';
	print '<table><tbody>';
	foreach( $instrument_names as $instr_name ) {

		print $header;

		$group = 'ps-'.strtolower( $instr_name );
		$group4auth = 'gid:'.$group;
		$experiments = $logbook->experiments_for_instrument( $instr_name );
		foreach( $experiments as $experiment ) {
			$has_role = $authdb->hasRole( $group4auth, $experiment->id(), $application, $role );
			print
				'<tr>'.
				'<td class="table_cell table_cell_left">'.$instr_name.'</td>'.
				'<td class="table_cell">'.$experiment->id().'</td>'.
			    '<td class="table_cell"><a href="../portal/?exper_id='.$experiment->id().'">'.$experiment->name().'</a></td>'.
				'<td class="table_cell"><a href="../authdb/?action=view_group&gid='.$experiment->POSIX_gid().'">'.$experiment->POSIX_gid().'</a></td>'.
				'<td class="table_cell"><a href="../authdb/?action=view_group&gid='.$group.'">'.$group.'</a></td>'.
				'<td class="table_cell">'.$role.'</td>';
			if( $fix ) {
				if( !$has_role ) {
					$authdb->createRolePlayer( $application, $role, $experiment->id(), $group4auth );
					print
						'<td class="table_cell"><span style="color:green;">Yes</span></td>'.
						'<td class="table_cell table_cell_right"><span style="color:green; font-weight:bold;">just fixed</span></td>';
				} else {
					print
						'<td class="table_cell">Yes</td>'.
						'<td class="table_cell table_cell_right"></td>';
				}
			} else {
				print
					'<td class="table_cell table_cell_right">'.($has_role ? 'Yes' : '<span style="color:red;">No</span>').'</td>';
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