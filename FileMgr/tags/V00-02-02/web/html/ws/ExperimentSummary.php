<?php

require_once( 'regdb/regdb.inc.php' );
require_once( 'filemgr/filemgr.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use RegDB\RegDB;
use RegDB\RegDBAuth;
use RegDB\RegDBHtml;
use RegDB\RegDBException;

use FileMgr\FileMgrIrodsWs;
use FileMgr\FileMgrException;

use LusiTime\LusiTime;

/*
 * This script will process a request for displaying a status of an experiment.
 */
if( isset( $_GET['id'] )) {
    $id = trim( $_GET['id'] );
    if( $id == '' )
        die( "experiment identifier can't be empty" );
} else
    die( "no valid experiment identifier" );


/* Proceed with the operation
 */
try {
    RegDB::instance()->begin();

    $experiment = RegDB::instance()->find_experiment_by_id( $id )
        or die( "no such experiment" );

    $instrument = $experiment->instrument();

    if( !RegDBAuth::instance()->canRead( $experiment->id())) {
        print( RegDBAuth::reporErrorHtml(
            'You are not authorized to access any information about the experiment' ));
        exit;
    }

    /* Get the stats
     */
    $num_runs       = 0;
    $xtc_num_files  = 0;
    $xtc_size       = 0.0;
    $xtc_local_copy = 0;
    $xtc_archived   = 0;

    $hdf5_num_files  = 0;
    $hdf5_size       = 0.0;
    $hdf5_local_copy = 0;
    $hdf5_archived   = 0;

    $range = FileMgrIrodsWs::max_run_range( $instrument->name(), $experiment->name(), array('xtc','hdf5'));

    $num_runs      = $range['total'];
    $range_of_runs = $range['min'].'-'.$range['max'];
    $xtc_runs = null;
    FileMgrIrodsWs::runs( $xtc_runs, $instrument->name(), $experiment->name(), 'xtc', $range_of_runs );
    foreach( $xtc_runs as $run ) {
        $unique_files = array();  // per this run
        $files = $run->files;
        foreach( $files as $file ) {
            if( !array_key_exists( $file->name, $unique_files )) {
                $unique_files[$file->name] = $run->run;
                $xtc_num_files++;
                $xtc_size += $file->size / (1024.0 * 1024.0 * 1024.0);
            }
            if( $file->resource == 'hpss-resc'   ) $xtc_archived++;
            if( $file->resource == 'lustre-resc' ) $xtc_local_copy++;
        }
    }
    $xtc_size_str = sprintf( "%.0f", $xtc_size );
    $hdf5_runs = null;
    FileMgrIrodsWs::runs( $hdf5_runs, $instrument->name(), $experiment->name(), 'hdf5', $range_of_runs );
    foreach( $hdf5_runs as $run ) {
        $unique_files = array();  // per this run
        $files = $run->files;
        foreach( $files as $file ) {
            if( !array_key_exists( $file->name, $unique_files )) {
                $unique_files[$file->name] = $run->run;
                $hdf5_num_files++;
                $hdf5_size += $file->size / (1024.0 * 1024.0 * 1024.0);
            }
            if( $file->resource == 'hpss-resc'   ) $hdf5_archived++;
            if( $file->resource == 'lustre-resc' ) $hdf5_local_copy++;
        }
    }
    $hdf5_size_str = sprintf( "%.0f", $hdf5_size );

    header( 'Content-type: text/html' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

    $value_color = 'maroon';
    $label_color = '#b0b0b0';

    $elog_url =
        '<a href="../logbook?action=select_experiment_by_id&id='.$experiment->id().
        '" target="_blank" class="lb_link">LogBook</a>';

    $con = new RegDBHtml( 0, 0, 450, 180 );

    echo $con

        ->label (   0,   0, 'R u n s' )
        ->label (  20,  20, 'Number of runs:', false    )->value( 130, 20, $num_runs, $value_color )
                                                         ->value( 220, 20, '[ see '.$elog_url.' for details ]' )

        ->label (   0,  50, 'X T C' )
        ->label (  20,  70, 'Number of files:',  false )->value( 130,  70, $xtc_num_files,  $value_color )
        ->label ( 220,  70, 'Size [GB]:',        false )->value( 300,  70, $xtc_size_str,  $value_color )
        ->label (  20,  90, 'Archived to tape:', false )->value( 130,  90, $xtc_archived.' / '.$xtc_num_files, $value_color )
        ->label ( 220,  90, 'On disk:',          false )->value( 300,  90, $xtc_local_copy.' / '.$xtc_num_files, $value_color )

        ->label (   0, 120, 'H D F 5' )
        ->label (  20, 140, 'Number of files:',  false )->value( 130, 140, $hdf5_num_files,  $value_color )
        ->label ( 220, 140, 'Size [GB]:',        false )->value( 300, 140, $hdf5_size_str,  $value_color )
        ->label (  20, 160, 'Archived to tape:', false )->value( 130, 160, $hdf5_archived.' / '.$hdf5_num_files,   $value_color )
        ->label ( 220, 160, 'On disk:',          false )->value( 300, 160, $hdf5_local_copy.' / '.$hdf5_num_files, $value_color )
        ->html();

    RegDB::instance()->commit();

} catch (RegDBException   $e) { print $e->toHtml(); }
  catch (FileMgrException $e) { print $e->toHtml(); }

?>