<?php

/**
 * This service will obtain and return extended status of files found
 * in the request.
 */
require_once( 'authdb/authdb.inc.php' );
require_once( 'filemgr/filemgr.inc.php' );
require_once( 'logbook/logbook.inc.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

use FileMgr\FileMgrIrodsWs;
use FileMgr\FileMgrException;

use LogBook\LogBook;
use LogBook\LogBookException;

function report_error($msg) {
	return_result(
        array(
            'status' => 'error',
            'message' => $msg
        )
    );
}
function report_success($result) {
    $result['status'] = 'success';
  	return_result($result);
}
function return_result($result) {

	header( 'Content-type: application/json' );
	header( 'Cache-Control: no-cache, must-revalidate' ); // HTTP/1.1
	header( 'Expires: Sat, 26 Jul 1997 05:00:00 GMT' );   // Date in the past

    echo json_encode($result);
	exit;
}

try {

    $authdb = AuthDB::instance();
	$authdb->begin();

    $logbook = new LogBook();
    $logbook->begin();

    if( !isset($_POST['files'])) report_error("missing parameter: 'files'");
    $input_files = json_decode(trim($_POST['files']));
    if( is_null($input_files)) report_error("failed to translate parameter 'files' as a JSON object");
    if( !is_array($input_files)) report_error("expected array as parameter 'files'");

    // Group files by experiment and file type
    //
    $files_grouped = array();
    foreach($input_files as $triplet) {

        if( !is_array($triplet))
            report_error("expected array as a file description triplet in parameter 'files'");
        if( count($triplet) != 3)
            report_error("exactly 3 elements expected in file description triplets: (experiment_id,file_type,file_name)");

        $exper_id = intval($triplet[0]);
        if( !$exper_id ) report_error("invalid experiment identifier found in the request");
        if( !array_key_exists($exper_id, $files_grouped)) {
            $experiment = $logbook->find_experiment_by_id( $exper_id );
            if( is_null($experiment)) report_error("unknown experiment identifier {$exper_id} found in the request");
            $files_grouped[$exper_id] = array(
                'experiment' => $experiment,
                'files' => array()
            );
        }

        $file_type = strtolower($triplet[1]);
        if( !$file_type ) report_error("invalid file type '{$file_type}' found in the request");
        if( !array_key_exists($file_type, $files_grouped[$exper_id]['files'])) {
            $files_grouped[$exper_id]['files'][$file_type] = array();
        }
        array_push(
            $files_grouped[$exper_id]['files'][$file_type],
            $triplet[2]
        );
    }

    // Process the request according to the way the files were groupped
    // above.
    //
    $files_extended = array();
    foreach($files_grouped as $exper_id => $files_by_experiment) {

        $experiment = $files_by_experiment['experiment'];

        foreach( $files_by_experiment['files'] as $file_type => $files ) {


            // Obtain all files of the given type for the specified instrumet & experiment
            // from the migration database.
            //
            $data_migration_files = array();
            foreach( $experiment->regdb_experiment()->data_migration_files($file_type) as $file ) {
                $data_migration_files[$file->name()] = $file;
            }

            // Contact iRODS File Manager and obtain all files of the given type
            // for the specified instrument & experiment.
            //
            $files_in_irods = array(
                'lustre-resc' => array(),
                'hpss-resc'   => array()
            );
            {
                $range = FileMgrIrodsWs::max_run_range( $experiment->instrument()->name(), $experiment->name(), array($file_type));
                $range_of_runs = $range['min'].'-'.$range['max'];

                $runs = null;
                FileMgrIrodsWs::runs( $runs, $experiment->instrument()->name(), $experiment->name(), $file_type, $range_of_runs );
                foreach( $runs as $run ) {
                    foreach( $run->files as $file ) {
                        if( !array_key_exists($file->resource, $files_in_irods))
                            report_error ("implementation error: unknown storage resource reported by iRODS: '{$file->resource}'");
                        array_push(
                            $files_in_irods[$file->resource],
                            $file->name
                        );
                    }
                }
            }

            foreach( $files as $file_name ) {

                // Check if this file status in the File Manager and migration database.
                // Compute status flags accordingly.
                //
                $flags = array();
                if( array_key_exists      ($file_name, $data_migration_files))          array_push($flags,'IN_MIGRATION_DATABASE');
                if( False !== array_search($file_name, $files_in_irods['lustre-resc'])) array_push($flags,'DISK');
                if( False !== array_search($file_name, $files_in_irods['hpss-resc'  ])) array_push($flags,'HPSS');

                array_push(
                    $files_extended,
                    array(
                        array($exper_id,$file_type,$file_name),
                        $flags
                    )
                );
            }
        }
    }

	$authdb->commit();

    report_success(array('files_extended' => $files_extended));

} catch( AuthDBException  $e ) { report_error( $e->toHtml()); }
  catch( LogBookException $e ) { report_error( $e->toHtml()); }
  catch( FileMgrException $e ) { report_error( $e->toHtml()); }
  catch( Exception        $e ) { report_error( "{$e}" );      }


?>
