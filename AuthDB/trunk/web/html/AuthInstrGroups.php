<?php

require_once( 'authdb/authdb.inc.php' );
require_once( 'logbook/logbook.inc.php' );
require_once( 'regdb/regdb.inc.php' );

use AuthDB\AuthDB;

use LogBook\LogBook;

use RegDB\RegDB;

/* Harvest the optional parameters first.
 */
function report_error($msg) {
    print $msg;
    exit;
}
if( isset( $_GET['instr_name'] )) {
    $instr_name = strtoupper( trim( $_GET['instr_name'] ));
    if( $instr_name == '' )
        report_error( "<b>error:</b> instrument name parameter if present can't have an empty value" );
}
$fix = isset( $_GET['fix'] );

try {
    AuthDB::instance()->begin();

    if( $fix && !AuthDB::instance()->canEdit())
        report_error( "<b>error:</b> your account doesn't posseses sufficient privileges to perform the operaton" );

    LogBook::instance()->begin();

    $instrument_names = array();
    if( isset( $instr_name )) {
        if( is_null( LogBook::instance()->regdb()->find_instrument_by_name( $instr_name )))
            report_error( "<b>error:</b> the specified instrument isn't known" );
        array_push( $instrument_names, $instr_name  );
    } else {

        // Get all known instruments. Skip  pseudo-instruments.
        //
        foreach( LogBook::instance()->regdb()->instrument_names() as $name ) {
            $instrument = LogBook::instance()->regdb()->find_instrument_by_name($name);
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

body {
  margin: 0;
  padding: 0;
  font-family: Lucida Grande, Lucida Sans, Arial, sans-serif;
}
h2 {
  font-family: Lucida Grande, Lucida Sans, Arial, sans-serif;
}
p {
  font-size: 15px;
}
li {
  font-size: 14px;
}
table pre {
  margin: 0;
  padding: 0;
}
td.table_hdr {
  background-color:#d0d0d0;
  padding: 4px 8px 4px 8px;
  border: solid 1px #000000;
  border-top: none;
  border-left: none;
  /*
  font-family: Arial, sans-serif;
  */
  font-family: Lucida Grande, Lucida Sans, Arial, sans-serif;
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

<div style="padding:20px;">

  <h2>About</h2>
  <div style="padding-left:20px; max-width:720px;">
    <p>This script will check if instrument-specific POSIX groups are authorized as e-log 'Editor's of
       the corresponding experiments, and if the optional parameter <b>fix</b> is present then assign
       the role to the groups. Specifically, for each instrument 'XYZ', the tool will:</p>
    <ol>
      <li>check if there is such POSIX group as 'ps-xyz'</li>
      <li>check if the group is registered as e-log 'Editor' for all experiments of instrument 'XYZ'</li>
      <li>(optionally) if not then the group will be authorized for those experiments</li>
      <li>(optionally) report which changes which had to be made at step #3</li>
      <li>report the number of shifts for each experiment:
        <ul>
          <li>If the number is 0 then the number will be show in the red color because the older
              implementations of e-log won't be able to show runs reported by the DAQ system.</li>
          <li>The problem can be fixed by following a link to the Web Portal of such experiment
              (see a table column called 'Experiment') and starting the shift.</li>
        </ul>
    </ol>
    <p>The script has two optional parameters:</p>
    <ul>
      <li><b>instr_name='INSTRUMENT'</b> - limit a scope of the operation to the specified instrument</li>
      <li><b>fix</b> - proceed with the actual fix and authorize the groups. By default</li></ul>
  </div>
  <?php echo $extra_operations; ?>
  <h2>Authorizations</h2>
  <div style="padding-left:20px;">

<?php

    $elog_application = 'LogBook';
    $ldap_application = 'LDAP';

    print '<table><tbody>';
    foreach( $instrument_names as $instr_name ) {

        $experiments = LogBook::instance()->experiments_for_instrument( $instr_name );

        $instr_name_own    = $instr_name ;
        $instr_name_allied = $instr_name ;
        switch( $instr_name) {
            case 'XPP':
                $instr_name_allied = 'XCS';
                break;
            case 'XCS':
                $instr_name_allied = 'XPP';
                break;
        }
        if($instr_name_own != $instr_name_allied) {
            // The special case for XPP and XCS to allow cros-instrument E-log access
            // for members of each group to be e-log Editors of all experiments
            // in boths instruments.
            //
            $elog_role = 'Editor';
            $ldap_role = 'Admin';
            print
                '<tr>'.
                '  <td class="table_hdr" rowspan="3">Instrument</td>'.
                '  <td class="table_hdr" rowspan="3">Id</td>'.
                '  <td class="table_hdr" rowspan="3">Experiment</td>'.
                '  <td class="table_hdr" rowspan="3">User Group</td>'.
                '  <td class="table_hdr" rowspan="3"># shifts</td>'.
                '  <td class="table_hdr" colspan="7" align="center">OWN</td>'.
                '  <td class="table_hdr" colspan="7" align="center">ALLIED</td>'.
            ($fix ?
                '  <td class="table_hdr" rowspan="3">Action</td>' : '').
                '</tr>'.
                '<tr>'.
                '  <td class="table_hdr" colspan="3" align="center">E-Log</td>'.
                '  <td class="table_hdr" colspan="4" align="center">LDAP Admin</td>'.
                '  <td class="table_hdr" colspan="3" align="center">E-Log</td>'.
                '  <td class="table_hdr" colspan="4" align="center">LDAP Admin</td>'.
                '</tr>'.
                '  <td class="table_hdr">Group</td>'.
                '  <td class="table_hdr">Role</td>'.
                '  <td class="table_hdr">Is Authorized</td>'.
                '  <td class="table_hdr">Group</td>'.
                '  <td class="table_hdr">Is Authorized</td>'.
                '  <td class="table_hdr">Group</td>'.
                '  <td class="table_hdr">Is Authorized</td>'.
                '  <td class="table_hdr">Group</td>'.
                '  <td class="table_hdr">Role</td>'.
                '  <td class="table_hdr">Is Authorized</td>'.
                '  <td class="table_hdr">Group</td>'.
                '  <td class="table_hdr">Is Authorized</td>'.
                '  <td class="table_hdr">Group</td>'.
                '  <td class="table_hdr">Is Authorized</td>'.
                '<tr>'.
                '</tr>';

            $group_own           = 'ps-'.strtolower( $instr_name_own );
            $group4ldap_own      = $group_own.(in_array($instr_name_own, array('AMO','SXR','XPP','XCS','CXI','MEC')) ? '-sci' : '');
            $group4auth_own      = 'gid:'.$group_own;
            $group4auth4ldap_own = 'gid:'.$group4ldap_own;

            $group_allied           = 'ps-'.strtolower( $instr_name_allied );
            $group4ldap_allied      = $group_allied.(in_array($instr_name_allied, array('AMO','SXR','XPP','XCS','CXI','MEC')) ? '-sci' : '');
            $group4auth_allied      = 'gid:'.$group_allied;
            $group4auth4ldap_allied = 'gid:'.$group4ldap_allied;

            foreach( $experiments as $experiment ) {

                $comment = '';
                $num_shifts = $experiment->num_shifts();

                print
                    '<tr>'.
                    '<td class="table_cell table_cell_left">'.$instr_name.'</td>'.
                    '<td class="table_cell">'.$experiment->id().'</td>'.
                    '<td class="table_cell"><a href="../portal/?exper_id='.$experiment->id().'">'.$experiment->name().'</a></td>'.
                    '<td class="table_cell"><a href="../authdb/?action=view_group&gid='.$experiment->POSIX_gid().'">'.$experiment->POSIX_gid().'</a></td>'.
                    '<td class="table_cell"'.( $num_shifts == 0 ? ' style="font-color:red;"' : '' ).'>'.$num_shifts.'</td>';

                print
                    '<td class="table_cell"><a href="../authdb/?action=view_group&gid='.$group_own.'">'.$group_own.'</a></td>'.
                    '<td class="table_cell">'.$elog_role.'</td>';
                $has_elog_role_own = AuthDB::instance()->hasRole (
                    $group4auth_own, $experiment->id(), $elog_application, $elog_role) ;
                if ($fix) {
                    if( !$has_elog_role_own ) {
                        AuthDB::instance()->createRolePlayer (
                            $elog_application, $elog_role, $experiment->id(), $group4auth_own );
                        print
                            '<td class="table_cell"><span style="color:green;">Yes</span></td>';
                        $comment .= '<span style="color:green; font-weight:bold;">fixed '.$elog_application.' for OWN</span> ';
                    } else {
                        print
                            '<td class="table_cell">Yes</td>';
                    }
                } else {
                    print
                        '<td class="table_cell">'.($has_elog_role_own ? 'Yes' : '<span style="color:red;">No</span>').'</td>';
                }

                print
                    '<td class="table_cell"><a href="../authdb/?action=view_group&gid='.$group4ldap_own.'">'.$group4ldap_own.'</a></td>';
                $has_ldap_role_own = AuthDB::instance()->hasRole($group4auth4ldap_own,  $experiment->id(), $ldap_application, $ldap_role );
                if( $fix ) {
                    if( !$has_ldap_role_own ) {
                        AuthDB::instance()->createRolePlayer (
                            $ldap_application, $ldap_role, $experiment->id(), $group4auth4ldap_own );
                        print
                            '<td class="table_cell"><span style="color:green;">Yes</span></td>';
                        $comment .= '<span style="color:green; font-weight:bold;">fixed '.$ldap_application.' for OWN</span> ';
                    } else {
                        print
                            '<td class="table_cell">Yes</td>';
                    }
                } else {
                    print
                        '<td class="table_cell">'.($has_ldap_role_own ? 'Yes' : '<span style="color:red;">No</span>').'</td>';
                }
                $has_ldap_role_own = AuthDB::instance()->hasRole($group4auth_own, $experiment->id(), $ldap_application, $ldap_role );
                print
                    '<td class="table_cell"><a href="../authdb/?action=view_group&gid='.$group_own.'">'.$group_own.'</a></td>';
                if( $fix ) {
                    if( !$has_ldap_role_own ) {
                        AuthDB::instance()->createRolePlayer( $ldap_application, $ldap_role, $experiment->id(), $group4auth_own );
                        print
                            '<td class="table_cell"><span style="color:green;">Yes</span></td>';
                        $comment .= '<span style="color:green; font-weight:bold;">fixed '.$ldap_application.' for OWN</span> ';
                    } else {
                        print
                            '<td class="table_cell">Yes</td>';
                    }
                } else {
                    print
                        '<td class="table_cell">'.($has_ldap_role_own ? 'Yes' : '<span style="color:red;">No</span>').'</td>';
                }

                $has_elog_role_allied = AuthDB::instance()->hasRole($group4auth_allied, $experiment->id(), $elog_application, $elog_role );
                print
                    '<td class="table_cell"><a href="../authdb/?action=view_group&gid='.$group_allied.'">'.$group_allied.'</a></td>'.
                    '<td class="table_cell">'.$elog_role.'</td>';
                if( $fix ) {
                    if( !$has_elog_role_allied ) {
                        AuthDB::instance()->createRolePlayer($elog_application, $elog_role, $experiment->id(), $group4auth_allied );
                        print
                            '<td class="table_cell"><span style="color:green;">Yes</span></td>';
                        $comment .= '<span style="color:green; font-weight:bold;">fixed '.$elog_application.' for ALLIED</span> ';
                    } else {
                        print
                            '<td class="table_cell">Yes</td>';
                    }
                } else {
                    print
                        '<td class="table_cell">'.($has_elog_role_allied ? 'Yes' : '<span style="color:red;">No</span>').'</td>';
                }
                $has_ldap_role_allied = AuthDB::instance()->hasRole($group4auth4ldap_allied, $experiment->id(), $ldap_application, $ldap_role );
                print
                    '<td class="table_cell"><a href="../authdb/?action=view_group&gid='.$group4ldap_allied.'">'.$group4ldap_allied.'</a></td>';
                if( $fix ) {
                    if( !$has_ldap_role_allied ) {
                        AuthDB::instance()->createRolePlayer( $ldap_application, $ldap_role, $experiment->id(), $group4auth4ldap_allied );
                        print
                            '<td class="table_cell"><span style="color:green;">Yes</span></td>';
                        $comment .= '<span style="color:green; font-weight:bold;">fixed '.$ldap_application.' for ALLIED</span> ';
                    } else {
                        print
                            '<td class="table_cell">Yes</td>';
                    }
                } else {
                    print
                        '<td class="table_cell table_cell_right">'.($has_ldap_role_allied ? 'Yes' : '<span style="color:red;">No</span>').'</td>';
                }
                $has_ldap_role_allied = AuthDB::instance()->hasRole( $group4auth_allied, $experiment->id(), $ldap_application, $ldap_role );
                print
                    '<td class="table_cell"><a href="../authdb/?action=view_group&gid='.$group_allied.'">'.$group_allied.'</a></td>';
                if( $fix ) {
                    if( !$has_ldap_role_allied ) {
                        AuthDB::instance()->createRolePlayer( $ldap_application, $ldap_role, $experiment->id(), $group4auth_allied );
                        print
                            '<td class="table_cell"><span style="color:green;">Yes</span></td>';
                        $comment .= '<span style="color:green; font-weight:bold;">fixed '.$ldap_application.' for ALLIED</span> ';
                    } else {
                        print
                            '<td class="table_cell">Yes</td>';
                    }
                } else {
                    print
                        '<td class="table_cell table_cell_right">'.($has_ldap_role_allied ? 'Yes' : '<span style="color:red;">No</span>').'</td>';
                }
                
                if( $fix ) {
                    print
                        '<td class="table_cell table_cell_right">'.$comment.'</td>';
                }
                print
                    '</tr>';
            }
        } else {
            $elog_role = 'Editor';
            $ldap_role = 'Admin';
            print
                '<tr>'.
                '  <td class="table_hdr" rowspan="2">Instrument</td>'.
                '  <td class="table_hdr" rowspan="2">Id</td>'.
                '  <td class="table_hdr" rowspan="2">Experiment</td>'.
                '  <td class="table_hdr" rowspan="2">User Group</td>'.
                '  <td class="table_hdr" rowspan="2"># shifts</td>'.
                '  <td class="table_hdr" colspan="3" align="center">E-Log</td>'.
                '  <td class="table_hdr" colspan="4" align="center">LDAP Admin</td>'.
            ($fix ? '  <td class="table_hdr" rowspan="2">Action</td>' : '').
                '</tr>'.
                '  <td class="table_hdr">Group</td>'.
                '  <td class="table_hdr">Role</td>'.
                '  <td class="table_hdr">Is Authorized</td>'.
                '  <td class="table_hdr">Group</td>'.
                '  <td class="table_hdr">Is Authorized</td>'.
                '  <td class="table_hdr">Group</td>'.
                '  <td class="table_hdr">Is Authorized</td>'.
                '<tr>'.
                '</tr>';
            $group = 'ps-'.strtolower( $instr_name );
            $group4ldap = $group.(in_array($instr_name, array('AMO','SXR','XPP','XCS','CXI','MEC')) ? '-sci' : '');
            $group4auth = 'gid:'.$group;
            $group4auth4ldap = 'gid:'.$group4ldap;
            foreach( $experiments as $experiment ) {
                $comment = '';
                $num_shifts = $experiment->num_shifts();
                print
                    '<tr>'.
                    '<td class="table_cell table_cell_left">'.$instr_name.'</td>'.
                    '<td class="table_cell">'.$experiment->id().'</td>'.
                    '<td class="table_cell"><a href="../portal/?exper_id='.$experiment->id().'">'.$experiment->name().'</a></td>'.
                    '<td class="table_cell"><a href="../authdb/?action=view_group&gid='.$experiment->POSIX_gid().'">'.$experiment->POSIX_gid().'</a></td>'.
                    '<td class="table_cell"'.( $num_shifts == 0 ? ' style="font-color:red;"' : '' ).'>'.$num_shifts.'</td>'.
                    '<td class="table_cell"><a href="../authdb/?action=view_group&gid='.$group.'">'.$group.'</a></td>';
                $has_elog_role = AuthDB::instance()->hasRole( $group4auth, $experiment->id(), $elog_application, $elog_role );
                print
                    '<td class="table_cell">'.$elog_role.'</td>';
                if( $fix ) {
                    if( !$has_elog_role ) {
                        AuthDB::instance()->createRolePlayer( $elog_application, $elog_role, $experiment->id(), $group4auth );
                        print
                            '<td class="table_cell"><span style="color:green;">Yes</span></td>';
                        $comment .= '<span style="color:green; font-weight:bold;">fixed '.$elog_application.'</span> ';
                    } else {
                        print
                            '<td class="table_cell">Yes</td>';
                    }
                } else {
                    print
                        '<td class="table_cell">'.($has_elog_role ? 'Yes' : '<span style="color:red;">No</span>').'</td>';
                }
                $has_ldap_role = AuthDB::instance()->hasRole( $group4auth4ldap, $experiment->id(), $ldap_application, $ldap_role );
                print
                    '<td class="table_cell"><a href="../authdb/?action=view_group&gid='.$group4ldap.'">'.$group4ldap.'</a></td>';
                if( $fix ) {
                    if( !$has_ldap_role ) {
                        AuthDB::instance()->createRolePlayer( $ldap_application, $ldap_role, $experiment->id(), $group4auth4ldap );
                        print
                            '<td class="table_cell"><span style="color:green;">Yes</span></td>';
                        $comment .= '<span style="color:green; font-weight:bold;">fixed '.$ldap_application.'</span> ';
                    } else {
                        print
                            '<td class="table_cell">Yes</td>';
                    }
                } else {
                    print
                        '<td class="table_cell table_cell_right">'.($has_ldap_role ? 'Yes' : '<span style="color:red;">No</span>').'</td>';
                }
                $has_ldap_role = AuthDB::instance()->hasRole( $group4auth, $experiment->id(), $ldap_application, $ldap_role );
                print
                    '<td class="table_cell"><a href="../authdb/?action=view_group&gid='.$group.'">'.$group.'</a></td>';
                if( $fix ) {
                    if( !$has_ldap_role ) {
                        AuthDB::instance()->createRolePlayer( $ldap_application, $ldap_role, $experiment->id(), $group4auth );
                        print
                            '<td class="table_cell"><span style="color:green;">Yes</span></td>';
                        $comment .= '<span style="color:green; font-weight:bold;">fixed '.$ldap_application.'</span> ';
                    } else {
                        print
                            '<td class="table_cell">Yes</td>';
                    }
                } else {
                    print
                        '<td class="table_cell table_cell_right">'.($has_ldap_role ? 'Yes' : '<span style="color:red;">No</span>').'</td>';
                }
                if( $fix ) {
                    print
                        '<td class="table_cell table_cell_right">'.$comment.'</td>';
                }
                print
                    '</tr>';
            }
        }
    }
    print '</tbody><table>';
    
    LogBook::instance()->commit();
    AuthDB::instance()->commit();
    
} catch( Exception $e ) { report_error( $e.'<pre>'.print_r( $e->getTrace(), true ).'</pre>' ); }
  
?>

  </div>
</div>

</body>
</html>