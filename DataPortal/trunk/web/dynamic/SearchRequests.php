<?php

require_once('DataPortal/DataPortal.inc.php');
require_once('AuthDB/AuthDB.inc.php');
require_once('RegDB/RegDB.inc.php');
require_once('LogBook/LogBook.inc.php');
require_once('FileMgr/FileMgr.inc.php');

/*
 * This script will process requests for various information stored in the database.
 * The result will be returned an embedable HTML element (<div>).
 */
if( !isset( $_GET['exper_id'] )) die( "no valid experiment identifier in the request" );
$exper_id = trim( $_GET['exper_id'] );

/*
 * Oprtional parameter of the filter:
 *
 * runs       - a range of runs
 * translated - a flag indicating if the files were successfully translated (allowed values: '0', '1')
 * show_files - a flag indicating if files should be shown as well for each run
 */
if( isset( $_GET['runs'] )) {
    $range_of_runs = trim( $_GET['runs'] );
    if( $range_of_runs == '' )
        die( 'run range parameter shall not have an empty value' );
}

if( isset( $_GET['translated'] )) {
    $str = trim( $_GET['translated'] );
    if( $str == '1' ) $translated = true;
    else if( $str == '0' ) $translated = false;
    else {
        die( 'unsupported value of the translated parameter: '.$str );
    }
}

$show_files = isset( $_GET['show_files'] );

function pre( $str, $width=null, $color=null ) {
	$color_style = is_null( $color ) ? '' : ' style="color:'.$color.';"';
	if( is_null( $width )) return "<pre {$color_style}>".$str.'</pre>';
	return "<pre {$color_style}>".sprintf( "%{$width}s", $str ).'</pre>';
}

/**
 * Take an input dictionary of all runs and a range of runs to be used as
 * a filter, translate the range, walk through the input set of runs and
 * return only those which are found in the range. The result is returned
 * as a dictionary of the same kind as the input one. The later means that
 * whole kay-value pairs are carried over from the input to the resulting
 * dictionary.
 *
 * @param array $logbook_in - the dictionary with runs as keys and any type of values
 * @param string $range - a range of runs. It should
 * @return array  - of the same types as the input one
 */
function apply_filter_range2runs( $range, $logbook_in ) {

	/* Translate the range into a dictionary of runs. This is going to be
	 * our filter. Run numbers will be the keys. And each key will have True
	 * as the corresponding value.
	 */ 
	$runs2allow = array();
	foreach( explode( ',', $range ) as $subrange ) {

		/* Check if this is just a number or a subrange: <begin>-<end>
		 */
		$pair = explode( '-', $subrange );
		switch( count( $pair )) {
			case 1:
				$runs2allow[$pair[0]] = True;
				break;
			case 2:
				if( $pair[0] >= $pair[1] )
					die( "illegal subrange: ".$pair[0]."-".$pair[1] );
				for( $run = $pair[0]; $run <= $pair[1]; $run++ )
					$runs2allow[$run] = True;
				break;
			default:
				die( 'illegal syntax of the runs range' );
		}
	}


	/* Apply the filter
	 */
	$out = array();
	foreach( $logbook_in as $run ) {
		$runum = $run->num();
		if( array_key_exists( $runum, $runs2allow ))
			array_push( $out, $run );
	}

	return $out;
}

function apply_filter_translated2runs( $icws_in, $logbook_in, $require_translated ) {
	$out = array();
	foreach( $logbook_in as $run ) {
		$runum = $run->num();
		$translated = array_key_exists( $runum, $icws_in ) && ( $icws_in[$runum]->status == 'Complete' );
		if( $require_translated xor $translated ) continue;
		array_push( $out, $run );
	}
	return $out;
}

/* Turn two arrays (for XTC and HDF5 files) of objects into a dictionary of
 * objects keyed by run numbers. Each object in the resulting dictionary will
 * have a list of files of (either or both) XTC and HDF5 types.
 * 
 * NOTE: We require that each object in the input array has
 *       the following data members:
 *
 *         run   - the run number
 *         files - an array if files of the corresponding type
 */
function array2dict_and_merge( $in_xtc, $in_hdf5 ) {

	$out = array();
	foreach( $in_xtc as $i ) {
		$out[$i->run]['xtc']  = $i->files;
		$out[$i->run]['hdf5'] = null;
	}

	/* Note that not having XTC for a run is rathen unusual situation. But let's handle
	 * it at a higher level logic, not here. For now just put null to where the list
	 * of XTC files is expected.
	 */
	foreach( $in_hdf5 as $i ) {
		if( !array_key_exists( $i->run, $out )) {
			$out[$i->run]['xtc'] = null;
		}
		$out[$i->run]['hdf5'] = $i->files;
	}
	return $out;
}

function requests2dict( $in ) {
	$out = array();
	foreach( $in as $req ) $out[$req->run] = $req;
    return $out;
}

/**
 * Merge three dictionary and produce an output one. Apply optional filters
 * if requests. The filter is turned on if any filtering parameters are passed
 * to the script.
 *
 * @param array $icws_runs - the runs for which HDF5 translation attempts have even been made
 * @param array $irodsws_runs - the runs for which there are data files of any kind
 * @param array $logbook_runs - the primary source of runs which are known in a context fo teh experiment
 * @return array - see the code below for details
 */
function merge_and_filter( $icws_runs, $irodsws_runs, $logbook_runs_all ) {

	/* Apply two stages of optional filters first.
	 */
	global $range_of_runs;
	global $translated;

	$logbook_runs = isset( $range_of_runs ) ? apply_filter_range2runs( $range_of_runs, $logbook_runs_all ) : $logbook_runs_all;
	$logbook_runs = isset( $translated ) ? apply_filter_translated2runs( $icws_runs, $logbook_runs, $translated ) : $logbook_runs;

	$out = array();
	foreach( $logbook_runs as $run ) {
		$runnum = $run->num();
		array_push (
			$out,
			array (
				'logbook' => $run,
				'icws'    => ( array_key_exists( $runnum,    $icws_runs ) ?    $icws_runs[$runnum] : null ),
				'irodsws' => ( array_key_exists( $runnum, $irodsws_runs ) ? $irodsws_runs[$runnum] : null )
			)
		);
	}
	return $out;
}

/* Utility functions
 */
function simple_request_status( $status ) {
	switch( $status ) {

		case 'Initial_Entry':
		case 'Waiting_Translation':
			return 'queued';

		case 'Being_Translated':
			return 'translating';

		case 'Empty_Fileset':
		case 'H5Dir_Error':
		case 'Translation_Error':
		case 'Archive_Error':
			return 'failed';

		case 'Complete':
			return 'complete';
	}
   	return 'unknown';
}

function decorated_request_status( $status ) {
	$color = 'black';
    switch( $status ) {
    	case 'queued':		$color = '#c0c0c0'; break;
    	case 'translating':	$color = 'green'; break;
    	case 'failed':		$color = 'red'; break;
    }
    return '<span style="color:'.$color.'";">'.$status.'</span>';
}

/* -----------------------------
 * Begin the main algorithm here
 * -----------------------------
 */
try {

    $logbook = new LogBook();
    $logbook->begin();

    $experiment = $logbook->find_experiment_by_id( $exper_id )
        or die("No such experiment");

    $instrument = $experiment->instrument();

    header( 'Content-type: text/html' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past

    echo DataPortal::table_begin_html(
		array(
			array( 'name' => 'Run',        'width' =>  30 ),
			array( 'name' => 'End of Run', 'width' =>  90 ),
			array( 'name' => 'File',       'width' => 200 ),
			array( 'name' => 'Size',       'width' =>  90 ),
			array( 'name' => 'Status',     'width' =>  30 ),
			array( 'name' => 'Changed',    'width' =>  90 ),
			array( 'name' => 'Log',        'width' =>  30 ),
			array( 'name' => 'Priority',   'width' =>  40 ),
			array( 'name' => 'Actions',    'width' =>  40 ),
			array( 'name' => 'Comments',   'width' => 200 )
		)
	);

    /* Ignore this experiment if it hasn't started yet. Otherwise some of
     * the operations with Web services may cause troubles.
     */
    //if( $experiment->begin_time()->less( LusiTime::now())) {

   		$runs = merge_and_filter (
        	requests2dict (
        		FileMgrIfaceCtrlWs::experiment_requests (
            		$instrument->name(),
            		$experiment->name(),
            		true	/* one request per run, latest requests only */
           		)
           	),
        	array2dict_and_merge (
           		FileMgrIrodsWs::all_runs (
					$instrument->name(),
					$experiment->name(),
					'xtc'
				),
           		FileMgrIrodsWs::all_runs (
					$instrument->name(),
					$experiment->name(),
					'hdf5'
				)
			),
           	array_reverse( $experiment->runs())
    	);
    	foreach( $runs as $run ) {

    		$run_logbook = $run['logbook'];
    		$run_irodsws = $run['irodsws'];
            $run_icws    = $run['icws'];

            $runnum = $run_logbook->num();

    		$run_url    = pre( '<a class="link" href="/apps/logbook?action=select_run_by_id&id='.$run_logbook->id().'" target="_blank" title="click to see a LogBook record for this run">'.$runnum.'</a>' );
    		$end_of_run = pre( $run_logbook->end_time()->toStringShort());

    		$status_simple_if_available = is_null( $run_icws ) ? '' : simple_request_status( $run_icws->status );

    		$status   = '';
            $changed  = '';
            $logfile  = '';
            $priority = '';
            $actions  = '';
            $comments = '';
            
            /* The 'Translate' command will be enabled only for the runs for which File Manager
             * has HDF5 files while XTC files are already present.
             *
			 *   TODO: For now do not separate disk resident versus tape resident files!
			 *   Just cyheck if any replica of those files is available. Later we can
			 *   implement a smarter logic on how to display a status of files which
			 *   only exist on tape.
             */
            $xtc_files_found  = !is_null( $run_irodsws ) && !is_null( $run_irodsws['xtc' ] ) && count( $run_irodsws['xtc' ] );
            $hdf5_files_found = !is_null( $run_irodsws ) && !is_null( $run_irodsws['hdf5'] ) && count( $run_irodsws['hdf5'] );

            if(   $xtc_files_found && !$hdf5_files_found &&
            	( $status_simple_if_available != 'queued'      ) &&
            	( $status_simple_if_available != 'translating' ) &&
            	( $status_simple_if_available != 'complete'    )) {

            	$actions = '<button class="translate" style="font-size:12px;" value="'.$runnum.'">Translate</button>';
           	}

           	/* Make sure disk-resident replicas for all XTC files are available (as reported
           	 * by IRODS) before allowing translation. This step relies on optional "open file"
           	 * records posted by the DAQ system immediattely after creating the data files.
           	 *
           	 * TODO: this information may not exist for older experiments. Consider
           	 * fixing the data base by populating it with file creation timestamps.
           	 */
           	$files_open_by_DAQ = $experiment->regdb_experiment()->files( $runnum );
           	if( count( $files_open_by_DAQ )) {

           		$files_irodsws_num = 0;

           		$files_irodsws = $run_irodsws['xtc'];
           		if( !is_null( $files_irodsws ))
           			foreach( $files_irodsws as $f )
           				if( $f->replica == 0 ) $files_irodsws_num++;

          		if( $files_irodsws_num != count( $files_open_by_DAQ )) {
           			$actions = '';
           			$comments = 'only '.$files_irodsws_num.' out of '.count( $files_open_by_DAQ ).' XTC files available';
           		}
           	}

            /* The 'Elevate Priority' and 'Delete' commands are only available
             * for existing translation requests waiting in a queue. Also note,
             * that the priority number is also available for this type of requests.
             */
            if( $status_simple_if_available == 'queued' ) {
				$priority = '<pre id="priority_'.$run_icws->id.'">'.$run_icws->priority.'</pre>';
            	$actions .= '<button class="escalate" style="font-size:12px;" value="'.$run_icws->id.'">Escalate</button>'.
            				'<button class="delete" style="font-size:12px;" value="'.$run_icws->id.'">Delete</button>';
            }

            /* Note that the translation completion status for those runs for which
             * we do not have any data from the translation service is pre-determined
             * by a presence of HDF5 fiules. Moreover, of those files are present then
             * we _always_ assume that the translation succeeded regardeless of what
             * the translation service says (we're still going to show that info if available).
             * In case of a possible conflict when HDF5 are present but the translation service
             * record (if present) says something else, we just do not all any actions
             * on that file.
             */
            if( $hdf5_files_found ) {
            	$status = decorated_request_status( 'complete' );
            	if( !is_null( $run_icws )) {
            		$priority = '';
            		$actions = '';
            	}
            } else {
            	if( !is_null( $run_icws )) {
            		$status = decorated_request_status( simple_request_status( $run_icws->status ));
            	}
            }

            /* The status change timestamp is calculated based on the status.
             */
            switch( simple_request_status( $run_icws->status )) {
    			case 'queued':
    				$changed = pre( $run_icws->created );
    				break;
    			case 'translating':
		    		$changed = pre( $run_icws->started );
    				break;
    			case 'failed':
    			case 'complete':
    				$changed = pre( $run_icws->stopped );
    				break;
            }
            
            /* If the latest translation log file is available then alwasy show it.
             * It's up to a user to figure out if that file makes any sense for him/her.
             */
            if( !is_null( $run_icws ) && ( $run_icws->log_url != '' )) {
            	$id = $run_icws->id;
		    	$logfile = '<a class="link" href="translate/'.$id.'/'.$id.'.log" target="_blank" title="click to see the log file for the last translation attempt">log</a>';
            }

            /* Now show the very first summary line for that run. This may be the only
             * line in case if a user doesn't want to see the files, or if no XTC files
             * are present.
             */
            echo DataPortal::table_row_html(
    			array(
    				$run_url,
    				$end_of_run,
    				'',
    				'',
    				$status,
    				$changed,
    				$logfile,
    				$priority,
    				$actions,
    				$comments
   				),
   				!( $show_files && ( $xtc_files_found || $hdf5_files_found ))
    		);

    		/* Optionally (if requested) show the files of both types.
    		 */
    		if( $show_files && ( $xtc_files_found || $hdf5_files_found )) {

    			/* Separate production of rows from displaying them because we
    			 * don't want to have two passes through the lists of files to
    			 * calculate the 'end-of-group' requirement for the last row.
    			 */
            	$rows = array();
            	foreach( array( 'xtc', 'hdf5') as $type ) {

            		$files = $run_irodsws[$type];
            		if( is_null( $files )) continue;

            		foreach( $files as $f ) {

            			/* TODO: For now consider disk resident files only! Implement a smarter
            		 	 * login for files which only exist on HPSS. Probably show their status.
            			 */
            			if( $f->replica != 0 ) continue;

            			$color = $type == 'xtc' ? 'maroon' : null;
                		$name = pre( $f->name, null, $color );
                		$size = pre( number_format( $f->size ), 17, $color );    // less than 10 TB

                		array_push(
                			$rows,
    						array( '', '', $name, $size, '', '', '', '', '', ''	)
   						);
            		}
            	}

            	$filenum  = 0;
            	$numfiles = count( $rows );

            	foreach( $rows as $r ) {
		    		$filenum++;
                	$end_of_group = ( $filenum == $numfiles );
            		echo DataPortal::table_row_html( $r, $end_of_group );
            	}
    		}
    	}
    //}
    echo DataPortal::table_end_html();
    
} catch( AuthDBException $e ) {
	echo $e->toHtml();
} catch( RegDBException $e ) {
	echo $e->toHtml();
} catch( LogBookException $e ) {
	echo $e->toHtml();
} catch( FileMgrException $e ) {
	echo $e->toHtml();
}
?>