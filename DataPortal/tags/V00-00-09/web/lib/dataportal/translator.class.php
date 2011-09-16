<?php

namespace DataPortal;

require_once( 'dataportal/dataportal.inc.php' );
require_once( 'regdb/regdb.inc.php' );
require_once( 'logbook/logbook.inc.php' );
require_once( 'filemgr/filemgr.inc.php' );

use DataPortal\DataPortalexception;

use LogBook\LogBook;

use FileMgr\FileMgrIfaceCtrlWs;
use FileMgr\FileMgrIrodsWs;

/** The utility class to encapsulate services for implementing
 *  HDF5 translation applications.
 */
class Translator {

	static $types = array( 'xtc', 'hdf5');

	static function get_requests( $exper_id, $range_of_runs, $status, $show_files ) {

		$logbook = new LogBook();
	    $logbook->begin();

	    $experiment = $logbook->find_experiment_by_id( $exper_id );
	    if( is_null( $experiment ))
	    	throw new DataPortalexception(
            	__METHOD__,
                "No such experiment" );

        $instrument = $experiment->instrument();

	    $runs = Translator::merge_and_filter (
    	   	Translator::requests2dict (
       			FileMgrIfaceCtrlWs::experiment_requests (
           			$instrument->name(),
           			$experiment->name(),
	           		true	/* one request per run, latest requests only */
    	  		)
       		),
	       	Translator::array2dict_and_merge (
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
       		array_reverse( $experiment->runs()),
       		$range_of_runs,
       		$status
	    );

		$requests = array();

		foreach( $runs as $run ) {

    		$run_logbook = $run['logbook'];
    		$run_irodsws = $run['irodsws'];
			$run_icws    = $run['icws'];

	    	$status = Translator::simple_request_status( is_null( $run_icws ) ? '' : $run_icws->status );
    	    $actions = '';
        	$comments = '';
	        $ready4translation = false;

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

	        if( $xtc_files_found && !$hdf5_files_found &&
    	        !in_array( $status, array('FINISHED', 'TRANSLATING', 'QUEUED' ))) {
	        	$actions = '<button class="translate not4print" value="'.$run_logbook->num().'">Translate</button>';
    	    	$ready4translation = true;
        	}
        	/*
	        if( $xtc_files_found && $hdf5_files_found &&
    	        !in_array( $status, array('TRANSLATING', 'QUEUED' ))) {
	        	$actions = '<button class="translate not4print" value="'.$run_logbook->num().'">Retranslate</button>';
    	    	$ready4translation = true;
        	}
        	*/

	        /* Make sure disk-resident replicas for all XTC files are available (as reported
    	     * by IRODS) before allowing translation. This step relies on optional "open file"
	         * records posted by the DAQ system immediattely after creating the data files.
    	     *
        	 * TODO: this information may not exist for older experiments. Consider
	         * fixing the data base by populating it with file creation timestamps.
    	     */
	       	$files_open_by_DAQ = $experiment->regdb_experiment()->files( $run_logbook->num());
    	    if( count( $files_open_by_DAQ )) {

	        	$files_irodsws_num = 0;

	        	$files_irodsws = $run_irodsws['xtc'];
    	    	if( !is_null( $files_irodsws ))
        			foreach( $files_irodsws as $f )
        				if( $f->replica == 0 ) $files_irodsws_num++;

       				if( $files_irodsws_num != count( $files_open_by_DAQ )) {
	           			$actions = '';
    	       			$comments = 'only '.$files_irodsws_num.' out of '.count( $files_open_by_DAQ ).' XTC files available';
        	   			$ready4translation = false;
       				}
	       	}

	        /* The 'Elevate Priority' and 'Delete' commands are only available
    	     * for existing translation requests waiting in a queue. Also note,
        	 * that the priority number is also available for this type of requests.
	         */
    	    if( $status == 'QUEUED' ) {
	    	   	$actions =
	       			'<button class="escalate not4print" style="font-size:12px;" value="'.$run_icws->id.'">Escalate</button>'.
       				'<button class="stop not4print" style="font-size:12px;" value="'.$run_icws->id.'">Stop</button>';
	        }

	        /* Note that the translation completion status for those runs for which
	         * we do not have any data from the translation service is pre-determined
	         * by a presence of HDF5 files. Moreover, of those files are present then
	         * we _always_ assume that the translation succeeded regardeless of what
	         * the translation service says (we're still going to show that info if available).
	         * In case of a possible conflict when HDF5 are present but the translation service
	         * record (if present) says something else, we just do not all any actions
	         * on that file.
	         */
	        if( $hdf5_files_found && !is_null( $run_icws ) && ( $status == 'FAILED' )) {
    	       	$status = 'FINISHED';
        	}
	        if( !$hdf5_files_found && ( $status == 'FINISHED' )) {
    	       	$comments .= ($comments == '' ? '' : '; ')."If the translation just finished then HDF files<br>will appear shortly after they'll be archived to tape.";
        	}

	        /* The status change timestamp is calculated based on the status.
    	     */
        	$changed = '';
        	switch( $status ) {
   				case 'QUEUED':
   					$changed = !is_null( $run_icws ) && $run_icws->created ? $run_icws->created : '';
   					break;
	   			case 'TRANSLATING':
		    		$changed = !is_null( $run_icws ) && $run_icws->started ? $run_icws->started : '';
   					break;
   				case 'FAILED':
	   			case 'FINISHED':
   					$changed =  !is_null( $run_icws ) && $run_icws->stopped ? $run_icws->stopped : '';
   					break;
        	}
	        $request = array(
    	   		'state' => array(
   					'id' => $run_icws->id,
					'run_number' => $run_logbook->num(),
					'run_id' => $run_logbook->id(),
					'end_of_run' => is_null( $run_logbook->end_time()) ? '' : $run_logbook->end_time()->toStringShort(),
					'status' => $status,
					'changed' => $changed,
					'log_available' => ( !is_null( $run_icws ) && ( $run_icws->log_url != '' )) ? 1 : 0,
					'priority' => $status == 'QUEUED' ? $run_icws->priority : '',
					'actions' => $actions,
					'comments' => $comments,
        			'ready4translation' => $ready4translation ? 1 : 0
        		),
				'xtc' => array(),
				'hdf5' => array()
			);

	   		/* Optionally (if requested) show the files of both types.
   			 */
   			if( $show_files && ( $xtc_files_found || $hdf5_files_found )) {

	   			/* Separate production of rows from displaying them because we
   				 * don't want to have two passes through the lists of files to
   				 * calculate the 'end-of-group' requirement for the last row.
   				 */
	           	foreach( Translator::$types as $type ) {

	           		$files = $run_irodsws[$type];
	           		if( is_null( $files )) continue;

	           		foreach( $files as $f ) {

	           			/* TODO: For now consider disk resident files only! Implement a smarter
    	       		 	 * logic for files which only exist on HPSS. Probably show their status.
        	   			 */
           				if( $f->replica != 0 ) continue;

	               		array_push( $request[$type], array( 'name' => $f->name, 'size' => number_format( $f->size )));
    	       		}
        	   	}
	    	}
    		array_push( $requests, $request );
		}
		return $requests;
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
						throw new DataPortalexception(
            				__METHOD__,
		                	"illegal subrange: ".$pair[0]."-".$pair[1] );
					for( $run = $pair[0]; $run <= $pair[1]; $run++ )
						$runs2allow[$run] = True;
					break;
				default:
					throw new DataPortalexception(
            			__METHOD__,
                		'illegal syntax of the runs range' );
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

	function apply_filter_status2runs( $icws_in, $logbook_in, $required_status ) {
		$out = array();
		foreach( $logbook_in as $run ) {
			$runum = $run->num();
			$status = Translator::simple_request_status( array_key_exists( $runum, $icws_in ) ? $icws_in[$runum]->status : '' );
			if( $required_status != $status ) continue;
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
	 * @param array $logbook_runs - the primary source of runs which are known in a context fo the experiment
	 * @param string $range_of_runs - the optional filer for runs
	 * @param string $status - the optional filer for request status
	 * @return array - see the code below for details
	 */
	function merge_and_filter( $icws_runs, $irodsws_runs, $logbook_runs_all, $range_of_runs, $status ) {

		/* Apply two stages of optional filters first.
		 */
		$logbook_runs = isset( $range_of_runs ) ? Translator::apply_filter_range2runs( $range_of_runs, $logbook_runs_all ) : $logbook_runs_all;
		$logbook_runs = isset( $status ) ? Translator::apply_filter_status2runs( $icws_runs, $logbook_runs, $status ) : $logbook_runs;

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

	static function simple_request_status( $status ) {
		switch( $status ) {

			case 'Initial_Entry':
			case 'Waiting_Translation':
				return 'QUEUED';

			case 'Being_Translated':
				return 'TRANSLATING';

			case 'Empty_Fileset':
			case 'H5Dir_Error':
			case 'Translation_Error':
			case 'Archive_Error':
				return 'FAILED';

			case 'Complete':
				return 'FINISHED';
		}
   		return 'NOT-TRANSLATED';
	}
}
?>