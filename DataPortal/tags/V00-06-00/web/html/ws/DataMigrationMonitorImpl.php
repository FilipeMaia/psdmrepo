<?php

/* Parameters of the script
 */
$active_filter = isset( $_GET['active' ] );
$recent_filter = isset( $_GET['recent' ] );

$instrument_name_filter = null;
if( isset( $_GET['instrument' ] )) {
    $str = trim( $_GET['instrument' ] );
    if( $str != '' ) $instrument_name_filter = $str;
}

$skip_non_archived = isset( $_GET['skip_non_archived' ] );
$skip_non_local    = isset( $_GET['skip_non_local'    ] );
$skip_non_migrated = isset( $_GET['skip_non_migrated' ] );

$min_delay_sec = 0;
if     ( isset( $_GET['min_delay' ] )) $min_delay_sec = (int)trim( $_GET['min_delay' ] );

$ignore_older_than_seconds_ago = 0;
if     ( isset( $_GET['ignore_1h' ] )) $ignore_older_than_seconds_ago = 3600;
else if( isset( $_GET['ignore_1d' ] )) $ignore_older_than_seconds_ago = 24*3600;
else if( isset( $_GET['ignore_1w' ] )) $ignore_older_than_seconds_ago = 7*24*3600;

$notify = isset( $_GET['notify' ] );

require_once( 'authdb/authdb.inc.php' );
require_once( 'dataportal/dataportal.inc.php' );
require_once( 'logbook/logbook.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );
require_once( 'filemgr/filemgr.inc.php' );
require_once( 'regdb/regdb.inc.php' );

use AuthDB\AuthDB;

use DataPortal\Config;

use LogBook\LogBook;

use LusiTime\LusiTime;

use FileMgr\FileMgrIrodsWs;

use RegDB\RegDB;

class TableView {
    
    private $notify = false;
    private $notify_buffer = array();
    private $exper_id = 0;
    private $runnum = 0;

    public function __construct( $notify ) {
        $this->notify = $notify;
        if( $this->notify ) ;
        else                print <<<HERE
<table><tbody>

HERE;
    }
    public function __destruct() {
        if( $this->notify )    $this->notify_subscribers();
        else                print <<<HERE
</tbody></table>

HERE;
    }
    public function add_row( $experiment, $runnum, $run, $name, $type, $size, $created, $archived, $local, $delay ) {

        if( $this->notify ) {
            $operation = '';
            if     ( $archived == 'Yes' ) $operation = 'Lustre';
            else if( $local    == 'Yes' ) $operation = 'HPSS';
            else                          $operation = 'DAQ';
            array_push(
                $this->notify_buffer,
                array(
                    'exper_id'   => $experiment->id(),
                    'exper_name' => $experiment->name(),
                    'runnum'     => $runnum,
                    'file'       => $name,
                    'type'       => $type,
                    'operation'  => $operation,
                    'delay'      => $delay ));
        } else {
            $style4run = '';
            $style4file = '';

            $experiment_url = '';
            $run_url = '';

            if( $this->exper_id != $experiment->id()) {
                $this->exper_id = $experiment->id();
                $this->runnum = $runnum;

                echo <<<HERE
      <tr>
        <td class="table_hdr">Experiment</td>
        <td class="table_hdr">Run</td>
        <td class="table_hdr">File</td>
        <td class="table_hdr">Type</td>
        <td class="table_hdr">Size</td>
        <td class="table_hdr">Created</td>
        <td class="table_hdr">Archived</td>
        <td class="table_hdr">On disk</td>
        <td class="table_hdr">Delay ranges: 1hr, 1day, longer</td>
      </tr>

HERE;

                $style4run = $style4file = 'table_cell_top';

                $experiment_url =<<<HERE
<a href="../portal/?exper_id={$experiment->id()}" target="_blank">{$experiment->name()}</a>
HERE;
                $run_url = $run;

            } else {
                if( $this->runnum != $runnum ) {
                    $this->runnum = $runnum;
                    $run_url = $run;
                } else {
                    $style4run = 'table_cell_top';
                }
            }

            $max_width  = 60;
            $max_height = 16;
            $delay_html = '';
            if( $delay <= 3600 ) {
                $width = ceil( $max_width * ( $delay / 3600 ));
                $delay_html .=
                    '<div style="float:left; width:'.$width.'; height:'.$max_height.'; background-color:#008000;"></div>'.
                    '<div style="float:left; width:'.($max_width-$width-1).'; height:'.($max_height-2).'; border:1px solid #c0c0c0; border-left:none;"></div>'.
                    '<div style="float:left; width:'.($max_width-1).'; height:'.($max_height-2).'; border:1px solid #c0c0c0; border-left:none;"></div>'.
                    '<div style="float:left; width:'.($max_width-1).'; height:'.($max_height-2).'; border:1px solid #c0c0c0; border-left:none;"></div>';
            } else {
                if( $delay <= 24*3600 ) {
                    $width = ceil( $max_width * ( $delay / ( 24*3600 )));
                    $delay_html .=
                        '<div style="float:left; width:'.$max_width.'; height:'.$max_height.'; background-color:orange;"></div>'.
                        '<div style="float:left; width:'.$width.'; height:'.$max_height.'; background-color:orange;"></div>'.
                        '<div style="float:left; width:'.($max_width-$width-1).'; height:'.($max_height-2).'; border:1px solid #c0c0c0; border-left:none;"></div>'.
                        '<div style="float:left; width:'.($max_width-$width-1).'; height:'.($max_height-2).'; border:1px solid #c0c0c0; border-left:none;"></div>';
                } else {
                    $delay_html .=
                        '<div style="float:left; width:'.(3*$max_width).'; height:'.$max_height.'; background-color:red;"></div>';
                }
            }
            $delay_html .= '<div style="clear:both;"></div>';

            echo <<<HERE
      <tr>
        <td class="table_cell table_cell_left table_cell_top">{$experiment_url}</td>
        <td class="table_cell table_cell_left {$style4run}">{$run_url}</td>
        <td class="table_cell {$style4file}">{$name}</td>
        <td class="table_cell {$style4file}">{$type}</td>
        <td class="table_cell {$style4file}">{$size}</td>
        <td class="table_cell {$style4file}">{$created}</td>
        <td class="table_cell {$style4file}">{$archived}</td>
        <td class="table_cell {$style4file}">{$local}</td>
        <td class="table_cell table_cell_right table_cell_top">{$delay_html}</td>
      </tr>

HERE;
        }
    }
    
    public function notify_subscribers() {

        if( $this->notify ) {
            print LusiTime::now()->toStringShort().': ';
            if( !count( $this->notify_buffer )) {
                print "[ PASS ]\n";
            } else {
                print "[ ALERT ]\n";

                $config = Config::instance();
                $config->begin();

                $daq = '';
                $hpss = '';
                $lustre = '';
                foreach( $this->notify_buffer as $entry ) {
                    $seconds = (int)( $entry['delay'] );
                    $days    = floor( $seconds / ( 24*3600 ));
                    $seconds = $seconds % ( 24*3600 );
                    $hours   = floor( $seconds / 3600 );
                    $seconds = $seconds % 3600;
                    $minutes = floor( $seconds / 60 );
                    $seconds = $seconds % 60;
                    $delay_str = '';
                    if( $days ) $delay_str .= $days.' days ';
                    $delay_str .= sprintf( "%02d:%02d.%02d", $hours, $minutes, $seconds );
                    $file = '      '.$entry['exper_name'].' [id:'.$entry['exper_id'].'] run:'.$entry['runnum'].'  '.$entry['file'].'  DELAY: '.$delay_str."\n";
                    switch( $entry['operation'] ) {
                    case 'DAQ'   : $daq    .= $file; break;
                    case 'HPSS'  : $hpss   .= $file; break;
                    case 'Lustre': $lustre .= $file; break;
                    }
                }
                $msg = '';
                if( $daq    != '' ) $msg .= "    Files which were never migrated from DAQ or later deleted:\n\n".$daq."\n";
                if( $hpss   != '' ) $msg .= "    Files which are not archived to HPSS:\n\n".$hpss."\n";
                if( $lustre != '' ) $msg .= "    Files which are not found on Lustre:\n\n".$lustre."\n";
            
                $url = ($_SERVER[HTTPS] ? "https://" : "http://" ).$_SERVER['SERVER_NAME'].'/apps-dev/portal/DataMigrationMonitor';
                $note = <<<HERE

                             ** NOTE **

The message was sent by the automated notification system because this e-mail
has been found registered to recieve alerts on data migration delays within
PCDS DAQ and OFFLINE.

To unsubscribe from this service, please use the Data Migration Monitor app:

  {$url}

HERE;

                foreach( $config->get_all_subscribed4migration() as $entry ) {
                    print "\n  NOTIFY: ".$entry['address'];
                    $config->do_notify ( $entry['address'], "*** ALERT ***", $msg.$note );
                }
                print "\n\n  MESSAGE:\n\n".$msg;
                $config->commit();
            }
        }
    }
}

function add_qualified_files( &$files, $infiles, $type, $file2run ) {

    global $skip_non_archived,
           $skip_non_local,
           $min_delay_sec,
           $ignore_older_than_seconds_ago,
           $now;

    $result = array();

    foreach( $infiles as $file ) {

        if( $file->type == 'collection' ) continue;

        /* Skip files for which we don't know run numbers
         */
        if( !array_key_exists( $file->name, $file2run )) continue;
        
        if( !array_key_exists( $file->name, $result )) {
            $result[$file->name] =
                array (
                    'name'     => $file->name,
                    'run'      => $file2run[$file->name],
                    'type'     => $type,
                    'size'     => $file->size,
                    'created'  => $file->ctime,
                    'archived'      => '<span style="color:red;">No</span>',
                    'archived_flag' => false,
                    'local'         => '<span style="color:red;">No</span>',
                    'local_flag'    => false,
                    'delay'    => $now->sec - $file->ctime );
        }
        if( $file->resource == 'hpss-resc' ) {
            $result[$file->name]['archived'] = 'Yes';
            $result[$file->name]['archived_flag'] = true;
        }
        if( $file->resource == 'lustre-resc' ) {
            $result[$file->name]['local'] = 'Yes';
            $result[$file->name]['local_flag'] = true;
            $result[$file->name]['local_path'] = $file->path;
            $result[$file->name]['created'] = $file->ctime;
        }
    }

    /* Filter out result by eliminating files are present at HPSS
     * and local disk storage.
     */
    foreach( $result as $file ) {
        $eligible = false;
        if( !$skip_non_archived ) $eligible = $eligible || !$file['archived_flag'];
        if( !$skip_non_local    ) $eligible = $eligible || !$file['local_flag'];
        if( !$eligible ) continue;
        $delay = $now->sec - $file['created'];
        if( $min_delay_sec                 && ( $delay < $min_delay_sec                 )) continue;
        if( $ignore_older_than_seconds_ago && ( $delay > $ignore_older_than_seconds_ago )) continue;
        $files[$file['name']] = $file;
    }
}

$now = null;

try {
    $now = LusiTime::now();

    $authdb = AuthDB::instance();
    $authdb->begin();

    $regdb = RegDB::instance();
    $regdb->begin();

    $logbook = LogBook::instance();
    $logbook->begin();

    $table = new TableView( $notify );

    $types = array( 'xtc', 'hdf5' );

    /* Get requested experiments. Eliminate those which haven't taken any data yet.
     * Order experiments by a tiome when their last run was taken.
     */
    $experiments_by_names = array();
    $experiments = array();
    if( $active_filter ) {
        foreach( $regdb->instrument_names() as $instrument_name ) {
            if( $regdb->find_instrument_by_name($instrument_name)->is_location()) continue;
            $experiment_switch = $regdb->last_experiment_switch( $instrument_name );
            if( !is_null( $experiment_switch )) {
                $exper_id = $experiment_switch['exper_id'];
                $experiment = $logbook->find_experiment_by_id( $exper_id );
                if( is_null( $experiment ))
                    die( "fatal internal error when resolving experiment id={$exper_id} in the database" );
                if( !is_null( $instrument_name_filter ) && ( $experiment->instrument()->name() != $instrument_name_filter )) continue;
                if( is_null( $experiment->find_last_run())) continue;
                if( array_key_exists( $experiment->name(), $experiments_by_names )) continue;
                $experiments_by_names[$experiment->name()] = True;
                array_push( $experiments, $experiment );
            }
        }    
    }
    if( $recent_filter ) {
        foreach( $logbook->experiments() as $experiment ) {
            if( !is_null( $instrument_name_filter ) && ( $experiment->instrument()->name() != $instrument_name_filter )) continue;
            $last_run = $experiment->find_last_run();
            if( is_null( $last_run )) continue;
            if( $last_run->begin_time()->sec < $now->sec - 7*24*3600 ) continue;
            if( array_key_exists( $experiment->name(), $experiments_by_names )) continue;
            $experiments_by_names[$experiment->name()] = True;
            array_push( $experiments, $experiment );
        }
    }
    if( !$active_filter && !$recent_filter ) {
        foreach( $logbook->experiments() as $experiment ) {
            if( !is_null( $instrument_name_filter ) && ( $experiment->instrument()->name() != $instrument_name_filter )) continue;
            if( is_null( $experiment->find_last_run())) continue;
            if( array_key_exists( $experiment->name(), $experiments_by_names )) continue;
            $experiments_by_names[$experiment->name()] = True;
            array_push( $experiments, $experiment );
        }
    }
    function cmp_by_last_run( $e1, $e2 ) {
        return $e1->find_last_run()->begin_time()->to64() < $e2->find_last_run()->begin_time()->to64();
    }
    usort( $experiments, "cmp_by_last_run" );

    foreach( $experiments as $experiment ) {

        $num_rows_printed = 0;

        $first_run = $experiment->find_first_run();
        $last_run  = $experiment->find_last_run();

        if( is_null($first_run) || is_null( $last_run )) continue;

        $range_of_runs = $first_run->num().'-'.$last_run->num();

        /* Build two structures:
         * - a mapping from file names to the corresponding run numbers. This information will be shown in the GUI.
         * - a list fop all known files.
         */
        $files_reported_by_iRODS = array();
        $file2run = array();
        $files    = array();
        foreach( $types as $type ) {
            $runs = null;
            FileMgrIrodsWs::runs( $runs, $experiment->instrument()->name(), $experiment->name(), $type, $range_of_runs );
            if( !is_null( $runs ))
                foreach( $runs as $run ) {
                    foreach( $run->files as $file ) {
                        $file2run[$file->name] = $run->run;
                        $files_reported_by_iRODS[$file->name] = True;
                    }
                    add_qualified_files( $files, $run->files, $type, $file2run );
                }
        }

        /* Build a map in which run numbers will be keys and lists of the corresponding
         * file descriptions will be the values.
         */
        $files_by_runs = array();
        foreach( $files as $file ) {
            $runnum = $file['run'];
            if( !array_key_exists( $runnum, $files_by_runs )) {
                $files_by_runs[$runnum] = array();
            }
            array_push( $files_by_runs[$runnum], $file );
        }

        /* Postprocess the above created array to missing gaps for runs
         * with empty collections of files.
         */
        for( $runnum = $first_run->num(); $runnum <= $last_run->num(); $runnum++ ) {
               if( !array_key_exists( $runnum, $files_by_runs )) {
                   $files_by_runs[$runnum] = array();
            }
        }

        /* Report the findings.
         */
        $run_numbers = array_keys( $files_by_runs );
        rsort( $run_numbers, SORT_NUMERIC );
        foreach( $run_numbers as $runnum ) {

            $run = $experiment->find_run_by_num( $runnum );
               $run_url = is_null( $run ) ?
                   $runnum :
                   '<a class="link" href="/apps/logbook?action=select_run_by_id&id='.$run->id().'" target="_blank" title="click to see a LogBook record for this run">'.$runnum.'</a>';

               foreach( $files_by_runs[$runnum] as $file ) {

                   $delay = $now->sec - $file['created'];

                   $table->add_row (
                    $experiment,
                    $runnum,
                    $run_url,
                    $file['name'],
                    strtoupper( $file['type'] ),
                    number_format( $file['size'] ),
                    date( "Y-m-d H:i:s", $file['created'] ),
                    $file['archived'],
                    $file['local'],
                    $delay );
               }

               /* Add XTC files which haven't been reported to iRODS because they have either
                * never migrated from ONLINE or because theye have been permanently deleted.
                */
            if( !$skip_non_migrated ) {
                foreach( $experiment->regdb_experiment()->files( $runnum ) as $file ) {

                    $name = sprintf("e%d-r%04d-s%02d-c%02d.xtc",
                                    $experiment->id(),
                                    $file->run(),
                                    $file->stream(),
                                    $file->chunk());

                    if( !array_key_exists( $name, $files_reported_by_iRODS )) {

                        $delay = $now->sec - $file->open_time()->sec;
                        if( $min_delay_sec                 && ( $delay < $min_delay_sec                 )) continue;
                        if( $ignore_older_than_seconds_ago && ( $delay > $ignore_older_than_seconds_ago )) continue;

                        $table->add_row (
                            $experiment,
                            $runnum,
                            $run_url,
                            $name,
                            'XTC',
                            '<span style="color:red;">n/a</span>',
                            date( "Y-m-d H:i:s", $file->open_time()->sec ),
                            '<span style="color:red;">n/a</span>',
                            '<span style="color:red;">never migrated from DAQ or deleted</span>',
                            $delay );
                    }
                }
            }
        }
    }
    unset( $table );

    $regdb->commit();
    $logbook->commit();
    
} catch( Exception $e ) { print '<pre style="padding:10px; border-top:solid 1px maroon; color:maroon;">'.print_r($e,true).'</pre>'; }
  
?>
