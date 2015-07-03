<?php

require_once( 'authdb/authdb.inc.php' );
require_once( 'logbook/logbook.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );
require_once( 'filemgr/filemgr.inc.php' );
require_once( 'regdb/regdb.inc.php' );

require_once( 'pdf-php/class.ezpdf.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

use LogBook\LogBook;
use LogBook\LogBookException;

use LusiTime\LusiTime;
use LusiTime\LusiTimeException;

use FileMgr\FileMgrIfaceCtrlWs;
use FileMgr\FileMgrIrodsWs;
use FileMgr\FileMgrException;

use RegDB\RegDBException;

class Pdf {

	public static $red = array( 255,0,0 );
	public static $green = array( 0, 255, 0 );
	public static $blue = array( 0, 0, 255 );
	public static $black = array( 0, 0, 0 );
	public static $grey = array( 255, 255, 0 );
	public static $white = array( 255, 255, 255 );

	public static function default_color() { return Pdf::$black; }

	private $pdf = null;
	private $chapter = 1;
	private $section = 1;

	public function pdf() { return $this->pdf; }

	public function __construct( $titles, $subtitles=null, $report_type=null ) {

		$this->pdf = new Cezpdf('letter', 'portrait');
		//$this->pdf->ezSetMargins(20,20,20,20);

		$this->timestamp();
		$this->pdf->ezSetDY( -20 );

		foreach( $titles as $t )
			$this->title(
				$t['text'],
				array_key_exists( 'color', $t ) ? $t['color'] : null,
				32 );

		if( !is_null( $subtitles )) {
			$this->pdf->ezSetDY(-12);
			foreach( $subtitles as $t )
				$this->title(
					$t['text'],
					array_key_exists( 'color', $t ) ? $t['color'] : null,
					20 );
		}
		if( !is_null( $report_type )) {
			$this->pdf->ezSetDY(-24);
			$this->pdf->selectFont( './fonts/Times.afm' );
			$this->pdf->ezText( 'Report(s): ', 20, array( 'left' => 175 ));
			$this->pdf->ezSetDY(18);
			$this->pdf->selectFont( './fonts/Times-Italic.afm' );
			$this->set_color( Pdf::$blue );
			$this->pdf->ezText( $report_type, 16, array( 'left' => 275 ));
			$this->set_color( Pdf::default_color());
		}
		$this->pdf->ezSetDY(-32);
	}
	
	private function set_color( $color ) {
		$this->pdf->setColor( $color[0], $color[1], $color[2] );
		return $this;
	}

	private function timestamp() {
		$font_size=10;
		$this->pdf->selectFont( './fonts/Times-Italic.afm' );
		$this->set_color( Pdf::default_color());
		$this->pdf->ezText(
			'generated on '.LusiTime::now()->toStringShort().
			' by '.AuthDB::instance()->authName(),
			$font_size,
			array( 'justification' => 'right' )
		);
		return $this;
	}

	private function title( $text, $color, $font_size ) {
		$this->pdf->selectFont( './fonts/Times-Bold.afm' );
		if( !is_null( $color )) $this->set_color( $color );
		$this->pdf->ezText( $text, $font_size, array( 'justification' => 'center' ));
		if( !is_null( $color )) $this->set_color( Pdf::default_color());
		return $this;
	}

	public function chapter( $text, $color=null ) {
		$font_size=20;
		$this->pdf->selectFont( './fonts/Times-Bold.afm' );
		if( !is_null( $color )) $this->set_color( $color );
		$this->pdf->ezText( $this->chapter++.'. '.$text, $font_size, array( 'justification' => 'left' ));
		if( !is_null( $color )) $this->set_color( Pdf::default_color());
		$this->pdf->ezSetDY(-8);
		$this->section=1;	// reset section number
		return $this;
	}

	public function section( $text, $color=null ) {
		$font_size=16;
		$this->pdf->selectFont( './fonts/Times-Bold.afm' );
		if( !is_null( $color )) $this->set_color( $color );
		$this->pdf->ezText( $this->chapter.'.'.$this->section++.' '.$text, $font_size, array( 'justification' => 'left' ));
		if( !is_null( $color )) $this->set_color( Pdf::default_color());
		$this->pdf->ezSetDY(-8);
		return $this;
	}
	
	public function paragraph( $text, $color=null ) {
		$font_size=12;
		$this->pdf->selectFont( './fonts/Times-Roman.afm' );
		if( !is_null( $color )) $this->set_color( $color );
		$this->pdf->ezText( '        '.$text, $font_size, array( 'justification' => 'full' ));
		if( !is_null( $color )) $this->set_color( Pdf::default_color());
		$this->pdf->ezSetDY(-$font_size/2);
		return $this;
	}

	public function def( $term, $term_color, $definition ) {
		$font_size=12;
		$this->pdf->selectFont( './fonts/Times-Roman.afm' );
		$this->set_color( $term_color );
		$this->pdf->ezText( $term, $font_size, array( 'left' => 20 ));
		$this->set_color( Pdf::default_color());
		$this->pdf->ezSetDY( $font_size );
		$this->pdf->ezText( $definition, $font_size, array( 'left' => 80 ));
		$this->pdf->ezSetDY(-$font_size/2);
		return $this;
	}

	public function listitem( $text ) {
		$font_size=12;
		$this->pdf->selectFont( './fonts/Times-Roman.afm' );
		$this->pdf->ezText( ' - '.$text, $font_size, array( 'left' => 20 ));
		$this->pdf->ezSetDY(-$font_size/2);
		return $this;
	}

	public function stream() {
		$this->pdf->ezStream();
		return $this;
	}
}

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

    LogBook::instance()->begin();

    $experiment = LogBook::instance()->find_experiment_by_id( $exper_id ) or die("No such experiment");
    $instrument = $experiment->instrument();

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

    // Compute counters

    $total_runs = 0;
    $total_queued = 0;
    $total_translating = 0;
    $total_failed = 0;
    $total_complete = 0;
    $total_unknown = 0;

    foreach( $runs as $run ) {

    	$run_logbook = $run['logbook'];
    	$run_irodsws = $run['irodsws'];
        $run_icws    = $run['icws'];

        $total_runs++;
        if( !is_null( $run_icws )) {
        	switch( simple_request_status( $run_icws->status )) {
    			case 'queued':
    				$total_queued++;
    				break;
    			case 'translating':
		    		$total_translating++;
    				break;
    			case 'failed':
    				$total_failed++;
    				break;
    			case 'complete':
    				$total_complete++;
    				break;
    			default:
    				$total_unknown++;
        	}
        }
    }

    // Begin generating the document

    $pdf = new Pdf(
		array(
			array( 'text' => 'Data Portal of Experiment:' ),
			array( 'text' => $instrument->name().' / '.$experiment->name(), 'color' => Pdf::$blue )
		),
		array(
			array( 'text' => '[ id='.$experiment->id().' ]' )
		),
		'HDF5 Translation Status'
	);

	$pdf->chapter (
		'HDF5 Translation Status' )
		->section (
			'Introduction' )
			->paragraph (
				'This is the full report including all runs available at a time when the report was made. '.
				'The report includes all stages of the translation. '.
				'Explanation of the translation status values:' )
			->def (
				'queued', Pdf::$grey,
				'the translation request was submitted, and it is sitting in a queue waiting '.
				'till suffient resources will be made available. Note, that in some cases the requests '.
				'may not be being processed because the translation is not enable for this particular experiment. '.
				'A timestamps in the "Changed" collumn of the table corresponds to the time when the translation '.
				'request was submitted. Queued requests will be translated in an order defined by the requests '.
				'priority. The priority ranges from 0 to the maximal positive 32-bit integer.' )
			->def (
				'translating', Pdf::$green,
				'the request is being translated. The log file for the translation process should be available. '.
				'The file will be regularily updated. The "Changed" collumn of the table will lcorrespond '.
				'to the time when the translation started.' )
			->def (
				'failed', Pdf::$red,
				'the translation has failed. The details can be found in the supplied log file. The "Changed" '.
				'timestamp corresponds to the time when the translation aborted. ' )
			->def (
				'completed', Pdf::$black,
				'the translation succeeded and HDF5 files are available. The "Changed" timestamp corresponds to '.
				'the time when the translation finished. The details can be found in the log file.' )
		->section (
			'Summary' )
			->listitem (
				'The total number of runs: <b>NNN</b>' )
			->listitem(
				"Queued: <b>{$total_queued}</b>" )
			->listitem(
				"Being translated: <b>{$total_translating}</b>" )
			->listitem(
				"Failed: <b>{$total_failed}</b>" )
			->listitem(
				"Complete: <b{$total_complete}</b>" )
			->listitem(
				"Unknown state: <b>{$total_unknown}</b>" )
		->section (
			'Translation Requests' )
			->paragraph(
				'' );

	$table_data = array();
	
    foreach( $runs as $run ) {

    	$run_logbook = $run['logbook'];
    	$run_irodsws = $run['irodsws'];
        $run_icws    = $run['icws'];

        $runnum = $run_logbook->num();

   		$end_of_run = $run_logbook->end_time()->toStringShort();

   		$status_simple_if_available = is_null( $run_icws ) ? '' : simple_request_status( $run_icws->status );

   		$status   = '';
        $changed  = '';
        $priority = '';
            
        $xtc_files_found  = !is_null( $run_irodsws ) && !is_null( $run_irodsws['xtc' ] ) && count( $run_irodsws['xtc' ] );
        $hdf5_files_found = !is_null( $run_irodsws ) && !is_null( $run_irodsws['hdf5'] ) && count( $run_irodsws['hdf5'] );

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
       	}

        /* The request priority make sense only for those requests which are sitting
         * in the input queue.
         */
        if( $status_simple_if_available == 'queued' ) {
			$priority = $run_icws->priority;
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
           	$status = 'complete';
            if( !is_null( $run_icws )) {
            	$priority = '';
            }
        } else {
           	if( !is_null( $run_icws )) {
           		$status = simple_request_status( $run_icws->status );
           	}
        }

        /* The status change timestamp is calculated based on the status.
         */
        switch( simple_request_status( $run_icws->status )) {
    		case 'queued':
    			$changed = $run_icws->created;
    			break;
    		case 'translating':
		    	$changed = $run_icws->started;
    			break;
    		case 'failed':
    		case 'complete':
    			$changed = $run_icws->stopped;
    			break;
        }
            
        /* If the latest translation log file is available then alwasy show it.
         * It's up to a user to figure out if that file makes any sense for him/her.
         */
        if( !is_null( $run_icws ) && ( $run_icws->log_url != '' )) {
        	$id = $run_icws->id;
        }

        /* Now show the very first summary line for that run. This may be the only
         * line in case if a user doesn't want to see the files, or if no XTC files
         * are present.
         */
		array_push( $table_data,
			array(
				'Run'        => $runnum,
				'End of Run' => $end_of_run,
				'File'       => '',
				'Size'       => '',
				'Status'     => $status,
				'Changed'    => $changed,
				'Priority'   => $priority
			)
		);

   		/* Optionally (if requested) show the files of both types.
   		 */
   		if( $show_files && ( $xtc_files_found || $hdf5_files_found )) {

           	$rows = array();
           	foreach( array( 'xtc', 'hdf5') as $type ) {

           		$files = $run_irodsws[$type];
           		if( is_null( $files )) continue;

           		foreach( $files as $f ) {

           			/* TODO: For now consider disk resident files only! Implement a smarter
           		 	 * login for files which only exist on HPSS. Probably show their status.
           			 */
           			if( $f->replica != 0 ) continue;

               		$name = $f->name;
               		$size = number_format( $f->size );

					array_push( $table_data,
						array(
							'Run'        => '',
							'End of Run' => '',
							'File'       => $name,
							'Size'       => $size,
							'Status'     => '',
							'Changed'    => '',
							'Priority'   => ''
						)
					);
            	}
    		}
    	}
    }
    $pdf->pdf()->ezTable( $table_data );
	$pdf->stream();    

} catch( AuthDBException   $e ) { echo $e->toHtml(); }
  catch( LogBookException  $e ) { echo $e->toHtml(); }
  catch( LusiTimeException $e ) { echo $e->toHtml(); }
  catch( FileMgrException  $e ) { echo $e->toHtml(); }
  catch( RegDBException    $e ) { echo $e->toHtml(); }
?>