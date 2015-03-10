<?php

namespace DataPortal ;

require_once 'dataportal/dataportal.inc.php' ;
require_once 'regdb/regdb.inc.php' ;
require_once 'logbook/logbook.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;
require_once 'filemgr/filemgr.inc.php' ;

use DataPortal\DataPortalexception ;

use LogBook\LogBook ;
use LusiTime\LusiTime ;

use FileMgr\FileMgrIrodsDb ;

define('KB', 1024.0) ;
define('MB', 1024.0 * KB) ;
define('GB', 1024.0 * MB) ;

function autoformat_size ($bytes) {
    $normalized = $bytes ;
    $format     = '%d' ;
    $units      = '' ;
    if      ($bytes < KB) {}
    else if ($bytes < MB) { $normalized = $bytes / KB; $format = $bytes < 10 * KB ? '%.1f' : '%d'; $units = 'KB' ; }
    else if ($bytes < GB) { $normalized = $bytes / MB; $format = $bytes < 10 * MB ? '%.1f' : '%d'; $units = 'MB' ; }
    else                  { $normalized = $bytes / GB; $format = $bytes < 10 * GB ? '%.1f' : '%d'; $units = 'GB' ; }
    return sprintf($format, $normalized).' <span style="font-weight:bold; font-size:9px;">'.$units.'</span>' ;
}

function time2string ($t) {
    return is_null($t) ? '' : $t->toStringDay().'&nbsp;&nbsp;&nbsp;'.$t->toStringHMS() ;
}

/** The utility class to encapsulate services for implementing
 *  HDF5 translation applications.
 */
class Translator1 {

    static function get_requests ($ifacectrlws, $exper_id, $range_of_runs, $status) {

        LogBook::instance()->begin() ;
        FileMgrIrodsDb::instance()->begin() ;

        $experiment = LogBook::instance()->find_experiment_by_id($exper_id) ;
        if (is_null($experiment))
            throw new DataPortalexception(__METHOD__, "No such experiment") ;

        $instrument = $experiment->instrument() ;

        $runs = Translator1::merge_and_filter (

           Translator1::requests2dict (
                $ifacectrlws->experiment_requests (
                    $instrument->name() ,
                    $experiment->name() ,
                    true    /* one request per run, latest requests only */)) ,

           Translator1::array2dict_and_merge (
                FileMgrIrodsDb::instance()->runs (
                    $instrument->name() ,
                    $experiment->name() ,
                    'xtc')) ,

            array_reverse($experiment->runs()) ,

            $range_of_runs ,
            $status
        ) ;

        $requests = array() ;

        foreach ($runs as $run) {

            $run_logbook = $run['logbook'] ;
            $run_irodsws = $run['irodsws'] ;
            $run_icws    = $run['icws'] ;

            $status   = Translator1::simple_request_status(is_null($run_icws) ? '' : $run_icws->status) ;
            $actions  = '' ;
            $comments = '' ;
            $ready4translation = false ;

            switch ($status) {
                case 'NOT-TRANSLATED' :
                case 'FAILED' :
                    $actions = '<button class="control-button translate" name="translate" value="'.$run_logbook->num().'">TRANSLATE</button>' ;
                    $ready4translation = true ;
                    break ;
                case 'FINISHED' :
                    $actions = '<button class="control-button translate retranslate" name="translate" value="'.$run_logbook->num().'">RE-TRANSLATE</button>' ;
                    $ready4translation = true ;
                    break ;
                case 'QUEUED' :
                    $actions .= '<button class="control-button escalate" name="escalate" style="font-size:12px;" value="'.$run_icws->id.'">ESCALATE</button>' ;
                    $actions .= '<button class="control-button stop"     name="stop"     style="font-size:12px;" value="'.$run_icws->id.'">STOP</button>' ;
                    break ;
            }

            /* Warn a user if not XTC files created by the DAQ are available
             * on disk as reported iRODS.
             */
            $files_open_by_DAQ = $experiment->regdb_experiment()->files($run_logbook->num()) ;
            if (count($files_open_by_DAQ)) {

                $files_irodsws_num = 0;

                $files_irodsws = $run_irodsws['xtc'];
                if (!is_null($files_irodsws)) {
                    foreach ($files_irodsws as $f) {
                        if ($f->resource == 'lustre-resc') $files_irodsws_num++;
                    }
                }
                if ($files_irodsws_num != count($files_open_by_DAQ)) {
                    $comments = $files_irodsws_num.' / '.count($files_open_by_DAQ).' XTC files on disk';
                }
            }

            /* The status change timestamp depends on on the status.
             */
            $changed = null ;
            switch ($status) {
                case 'QUEUED' :
                    $changed = !is_null($run_icws) && $run_icws->created ? $run_icws->created : '' ;
                    break ;
                case 'TRANSLATING':
                    $changed = !is_null($run_icws) && $run_icws->started ? $run_icws->started : '' ;
                    break;
                case 'FAILED' :
                case 'FINISHED' :
                    $changed =  !is_null($run_icws) && $run_icws->stopped ? $run_icws->stopped : '' ;
                    break ;
            }
            $request = array (
                'state' => array (
                    'id'                => !is_null($run_icws) ? $run_icws->id : 0 ,
                    'run_number'        => $run_logbook->num() ,
                    'run_id'            => $run_logbook->id() ,
                    'end_of_run'        => time2string($run_logbook->end_time()) ,
                    'status'            => $status ,
                    'changed'           => time2string($changed ? LusiTime::parse($changed) : null) ,
                    'log_available'     => (!is_null($run_icws) && (isset($run_icws->log_url) && ($run_icws->log_url != ''))) ? 1 : 0 ,
                    'priority'          => $status == 'QUEUED' ? $run_icws->priority : '' ,
                    'actions'           => $actions ,
                    'comments'          => $comments ,
                    'ready4translation' => $ready4translation ? 1 : 0
                ) ,
                
                /* Total size for the XTC files */

                'dataset' => array (
                    'xtc'  => array (
                        'num_files'  => 0,
                        'size_auto'  => '' ,    // using the most appropriate scale
                        'size_bytes' => 0  ,    // as a number of bytes
                        'size_kb'    => '' ,
                        'size_mb'    => '' ,
                        'size_gb'    => '' ,
                        'size'       => ''      // comma separated by three decimals
                    )
                )

            ) ;

            $xtc_files_found = !is_null($run_irodsws) && !is_null($run_irodsws['xtc' ]) && count($run_irodsws['xtc' ]) ;
            if ($xtc_files_found) {

                $files = $run_irodsws['xtc'] ;
                if (is_null($files)) continue ;

                $num_files = 0 ;
                $bytes     = 0 ;

                foreach ($files as $f) {

                    /* TODO: For now consider disk resident files only! Implement a smarter
                     * logic for files which only exist on HPSS. Probably show their status.
                     */
                    if ($f->resource != 'lustre-resc') continue ;

                    $num_files++ ;
                    $bytes = $bytes + intval($f->size) ;
                }
                $request['dataset']['xtc']['num_files']  = $num_files ;
                $request['dataset']['xtc']['size_auto']  = autoformat_size($bytes) ;
                $request['dataset']['xtc']['size_bytes'] = $bytes ;
                $request['dataset']['xtc']['size_kb']    = sprintf($bytes < 10 * KB ? "%.1f" : "%d", $bytes / KB) ;
                $request['dataset']['xtc']['size_mb']    = sprintf($bytes < 10 * MB ? "%.1f" : "%d", $bytes / MB) ;
                $request['dataset']['xtc']['size_gb']    = sprintf($bytes < 10 * GB ? "%.1f" : "%d", $bytes / GB) ;
                $request['dataset']['xtc']['size']       = number_format($bytes) ;
            }
            array_push($requests, $request) ;
        }
        return $requests ;
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
    function apply_filter_range2runs ($range, $logbook_in) {

        $out = array() ;

        $min_runnum = null ;
        $max_runnum = null ;
        
        foreach ($logbook_in as $run) {
            $runum = $run->num() ;
            $min_runnum = $min_runnum ? min(array($min_runnum, $runum)) : $runum;
            $max_runnum = $max_runnum ? max(array($max_runnum, $runum)) : $runum;
        }
        
        /* Proceed to teh filter only if the input list of runs isn't empty */

        if ($min_runnum && $max_runnum) {

            /* Translate the range into a dictionary of runs. This is going to be
             * our filter. Run numbers will be the keys. And each key will have True
             * as the corresponding value. */ 

            $runs2allow = array() ;
            foreach (explode(',', $range) as $subrange) {

                /* Check if this is just a number or a subrange: <begin>-<end>
                 */
                $pair = explode('-', $subrange) ;
                switch (count($pair)) {
                    case 1:
                        $runs2allow[$pair[0]] = True;
                        break;
                    case 2:
                        $begin_run = $min_runnum;
                        if ($pair[0]) {
                            $begin_run = intval($pair[0]) ;
                            if (!$begin_run) throw new DataPortalexception(__METHOD__, "illegal run '".$pair[0]."' subrange: '".$subrange."'") ;
                        }
                        $end_run = $max_runnum;
                        if ($pair[1]) {
                            $end_run = intval($pair[1]) ;
                            if (!$end_run) throw new DataPortalexception(__METHOD__, "illegal run '".$pair[1]."' subrange: '".$subrange."'") ;
                        }
                        if ($begin_run >= $end_run)    throw new DataPortalexception(__METHOD__, "illegal subrange: '".$subrange."'") ;
                        if ($begin_run <  $min_runnum) throw new DataPortalexception(__METHOD__, "non-existing begin run in subrange: '".$subrange."'") ;
                        if ($end_run   >  $max_runnum) throw new DataPortalexception(__METHOD__, "non-existing end run in subrange: '".$subrange."'") ;

                        for ($run = $begin_run; $run <= $end_run; $run++)
                            $runs2allow[$run] = True;
                        break;
                    default:
                        throw new DataPortalexception(
                            __METHOD__, 'illegal syntax of the runs range') ;
                }
            }

            /* Apply the filter */

            foreach ($logbook_in as $run) {
                $runum = $run->num() ;
                if (array_key_exists($runum, $runs2allow))
                    array_push($out, $run) ;
            }
        }
        return $out;
    }

    function apply_filter_status2runs ($icws_in, $logbook_in, $required_status) {
        $out = array() ;
        foreach ($logbook_in as $run) {
            $runum = $run->num() ;
            $status = Translator1::simple_request_status(array_key_exists($runum, $icws_in) ? $icws_in[$runum]->status : '') ;
            if ($required_status != $status) continue;
            array_push($out, $run) ;
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
    function array2dict_and_merge ($in_xtc) {

        $out = array() ;
        if ($in_xtc) {
            foreach ($in_xtc as $i) {
                $out[$i->run]['xtc']  = $i->files;
                $out[$i->run]['hdf5'] = null;
            }
        }
        return $out;
    }

    function requests2dict ($in) {
        $out = array() ;
        foreach ($in as $req) $out[$req->run] = $req;
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
    function merge_and_filter ($icws_runs, $irodsws_runs, $logbook_runs_all, $range_of_runs, $status) {

        /* Apply two stages of optional filters first.
         */
        $logbook_runs = !is_null($range_of_runs) ? Translator1::apply_filter_range2runs ($range_of_runs, $logbook_runs_all)  : $logbook_runs_all ;
        $logbook_runs = !is_null($status)        ? Translator1::apply_filter_status2runs($icws_runs, $logbook_runs, $status) : $logbook_runs ;

        $out = array() ;
        foreach ($logbook_runs as $run) {
            $runnum = $run->num() ;
            array_push (
                $out,
                array (
                    'logbook' => $run,
                    'icws'    => (array_key_exists($runnum,    $icws_runs) ?    $icws_runs[$runnum] : null),
                    'irodsws' => (array_key_exists($runnum, $irodsws_runs) ? $irodsws_runs[$runnum] : null)
                )
            ) ;
        }
        return $out ;
    }

    static function simple_request_status ($status) {
        switch ($status) {

            case 'WAIT_FILES':
            case 'WAIT':
            case 'PENDING':
                return 'QUEUED';

            case 'RUN':
            case 'SUSPENDED':
                return 'TRANSLATING';

            case 'DONE':
                return 'FINISHED';

            case 'FAIL':
            case 'FAIL_COPY':
            case 'FAIL_NOINPUT':
            case 'FAIL_MKDIR':
                return 'FAILED';

        }
        return 'NOT-TRANSLATED';
    }
}
?>