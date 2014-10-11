<?php

require_once( 'filemgr/filemgr.inc.php' );
require_once( 'logbook/logbook.inc.php' );
require_once( 'regdb/regdb.inc.php' );

use FileMgr\FileMgrIrodsWs;
use FileMgr\FileMgrException;

use LogBook\LogBook;
use LogBook\LogBookException;

use RegDB\RegDBHtml;

/*
 * This script will process requests for various information stored in the database.
 * The result will be returned an embedable HTML element (<div>).
 */
if( !isset( $_GET['exper_id'] )) die( "no valid experiment identifier in the request" );
$exper_id = trim( $_GET['exper_id'] );

/*
 * Oprtional parameter of the filter:
 *
 * runs     - a range of runs
 * type     - file type (allowed values 'xtc', 'hdf5'. The case doesn't matter.)
 * archived - a flag indicating if the files are archived (allowed values: '0', '1')
 * local    - a flag indicating if there is a local copy of files (allowed values: '0', '1')
 */
$range_of_runs=null;
if( isset( $_GET['runs'] )) {
    $range_of_runs = trim( $_GET['runs'] );
    if( $range_of_runs == '' )
        die( 'run range parameter shall not have an empty value' );
}

$types = null;
if( isset( $_GET['types'] )) {
    $types = explode( ',', strtolower( trim( $_GET['types'] )));
}
if( is_null( $types ) || !count( $types )) {
    $types = array( 'xtc', 'hdf5' );
}

$archived = null;
if( isset( $_GET['archived'] )) {
    $str = trim( $_GET['archived'] );
    if( $str == '1' ) $archived = true;
    else if( $str == '0' ) $archived = false;
    else {
        die( 'unsupported value of the archived parameter: '.$str );
    }
}

$local = null;
if( isset( $_GET['local'] )) {
    $str = trim( $_GET['local'] );
    if( $str == '1' ) $local = true;
    else if( $str == '0' ) $local = false;
    else {
        die( 'unsupported value of the local parameter: '.$str );
    }
}

/* This flag, if present, would tell the script to return a plain list
 * of selected file paths on a local disk for those files for which
 * this would apply.
 */
$import = isset( $_GET['import'] );

function pre( $str, $width=null ) {
    if( is_null( $width )) return '<pre>'.$str.'</pre>';
    return '<pre>'.sprintf( "%{$width}s", $str ).'</pre>';
}

function add_files( &$files, $infiles, $type, $file2run, $archived, $local ) {
    
    $result = array();

    foreach( $infiles as $file ) {

        if( $file->type == 'collection' ) continue;

        // Skip files for which we don't know run numbers
        //
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
                    'local_flag'    => false );    
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

    /* Filter out result by eliminating files which do not pass the archived and/or local
     * filter requirements
     */
    foreach( $result as $file ) {
        if(( !is_null( $archived ) && ( $archived ^ $file['archived_flag'] )) ||
           ( !is_null( $local )    && ( $local    ^ $file['local_flag'] ))) continue;
        $files[$file['name']] = $file;
    }
}

/*
 * Analyze and process the request
 */
try {

    LogBook::instance()->begin();

    $experiment = LogBook::instance()->find_experiment_by_id( $exper_id ) or die("No such experiment");
    $instrument = $experiment->instrument();

    /* If no specific run range is provided find out the one by probing all
     * known file types.
     */
    if( is_null( $range_of_runs )) {
        $range = FileMgrIrodsWs::max_run_range( $instrument->name(), $experiment->name(), array('xtc','hdf5'));
        $range_of_runs = $range['min'].'-'.$range['max'];
    }

    /* Build a mapping from file names to the corresponding run numbers.
     * This information will be shown in the GUI.
     */
    $file2run = array();
    foreach( $types as $type ) {
        $runs = null;
        FileMgrIrodsWs::runs( $runs, $instrument->name(), $experiment->name(), $type, $range_of_runs );
        foreach( $runs as $run ) {
            $files = $run->files;
            foreach( $files as $file ) {
                $file2run[$file->name] = $run->run;
            }
        }
    }

    /* Build a list fop all known files.
     */
    $files = array();
    foreach( $types as $type ) {
        $this_type_files = null;
        FileMgrIrodsWs::files( $this_type_files, '/psdm-zone/psdm/'.$instrument->name().'/'.$experiment->name().'/'.$type );
        add_files( $files, $this_type_files, $type, $file2run, $archived, $local );
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

    if( $import ) {

        header( 'Content-type: text/plain' );
        header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
        header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

        foreach( array_keys( $files_by_runs ) as $runnum )
            foreach( $files_by_runs[$runnum] as $file )
                if( $file['local_flag'] )
                    print $file['local_path']."\n";

    } else {

        header( 'Content-type: text/html' );
        header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
        header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past
    
        /* Count the number of files and runs as we need to determine a number of rows
         * in the RegDBHtml below.
         */
        $num_rows = count( $files ) + count( $files_by_runs );
    
        $con = new RegDBHtml( 0, 0, 850, $num_rows );
    
        $row = 0;
        foreach( array_keys( $files_by_runs ) as $runnum ) {
    
            $files = $files_by_runs[$runnum];
    
            $run = $experiment->find_run_by_num( $runnum );
            if( is_null( $run )) {
                /* TODO: Note, that LogBook may not have the most recent information about
                 * the runs. Consider using SciMD database in that case.
                 */
                $run_url = pre( $runnum );
            } else {
                $run_url = pre( '<a href="/apps/logbook?action=select_run_by_id&id='.$run->id().'" target="_blank" title="click to see a LogBook record for this run">'.$runnum.'</a>' );
            }
            $con->label( 5, $row, $run_url );
    
            foreach( $files as $file ) {
    
                $name = pre( $file['name'] );
                if( $file['local_flag'] ) {
                    $local = pre( '<a href="javascript:display_path('."'".$file['local_path']."'".')">path</a>' );
                } else {
                    $local = pre( $file['local'] );
                }
                $type     = pre( strtoupper( $file['type'] ));
                $size     = pre( number_format( $file['size'] ), 17 );    // less than 10 TB
                $created  = pre( date( "Y-m-d H:i:s", $file['created'] ));
                $archived = pre( $file['archived'] );
        
                $con->value(  60, $row, $name );
                $con->value( 330, $row, $type );
                $con->value( 400, $row, $size );
                $con->value( 555, $row, $created );
                $con->value( 725, $row, $archived );
                $con->value( 805, $row, $local );
        
                $row += 20;
            }
            $con->line( 0, $row, 850, 'dashed', '1px', '#c0c0c0' );
            $row += 10;
        }
        print $con->html();
    }
    LogBook::instance()->commit();

} catch (LogBookException $e) { print $e->toHtml(); }
  catch (FileMgrException $e) { print $e->toHtml(); }

?>
