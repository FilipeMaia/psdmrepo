<?php

namespace SysMon ;

require_once 'sysmon.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

use \LusiTime\LusiTime ;

// Increase the default limit because the script may be required to process
// a lot of data.

ini_set('memory_limit', '512M') ;

define('KB', 1024.0) ;
define('MB', 1024.0 * KB) ;
define('GB', 1024.0 * MB) ;


/**
 * The base class for various implementations of file iterators
 */
abstract class FileItr implements \Iterator {

    // Object parameters

    protected $SVC = null ;

    // Cached for optimization

    private $now                 = null ;       // LusiTime::now()
    private $logbook_experiments = array() ;    // [exper_id]            -> LogBookExperiment
    private $instr_ids           = array() ;    // [exper_id]            -> instrument identifier
    private $instr_names         = array() ;    // [exper_id]            -> instrument name
    private $exper_names         = array() ;    // [exper_id]            -> experiment name
    private $irods_files         = array() ;    // [exper_id][run][file] -> list of file replicas

    // Iterator context

    private $itr           = null ;     // RegDBFileItr
    private $current_value = null ;     // stdClass

    /**
     * Constructor
     *
     * @param integer $exper_id
     * @param \DataPortal\ServiceJSON $SVC
     */
    public function __construct ($SVC) {
        $this->SVC = $SVC ;
        $this->now = LusiTime::now() ;
    }
    public function rewind () {
        $this->_init() ;
        $this->itr->rewind() ;
        $this->current_value = null ;
    }
    public function current () {
        $this->_init() ;

        // If this is the first current value request upon
        // the initialization or after moving to teh next element.

        if (is_null($this->current_value)) {
            $this->current_value = $this->_file2obj ($this->itr->current()) ;
        }
        return $this->current_value ;
    }
    public function key () {
        $this->_init() ;
        return $this->itr->key() ;
    }
    public function next () {
        $this->_init() ;
        $this->itr->next() ;
        $this->current_value = null ;   // will be dynamically cosstruted
    }
    public function valid () {
        $this->_init() ;
        return $this->itr->valid() ;
    }

    /**
     * Request a proper iterator from a subclass if needed
     * 
     * NOTE: This method is expected to be implemented
     *       by derived classes
     */
    protected abstract function get_itr () ;

    /**
     * Make sure the iteration context is initialized and the optimization
     * case is properly set.
     */
    private function _init () {
        if (is_null($this->itr)) $this->itr = $this->get_itr() ;
    }

    /**
     * Lazy loader for instrument identifiers
     *
     * @param \RegDB\RegDBFile $f
     * @return integer
     */
    private function _instr_id ($f) {
        $f_exper_id = $f->exper_id() ;
        if (!array_key_exists($f_exper_id, $this->instr_ids)) {
            $this->instr_ids[$f_exper_id] = $f->experiment()->instrument()->id() ;
        }
        return $this->instr_ids[$f_exper_id] ;
    }

    /**
     * Lazy loader for instrument names
     * 
     * @param \RegDB\RegDBFile $f
     * @return string
     */
    private function _instr_name ($f) {
        $f_exper_id = $f->exper_id() ;
        if (!array_key_exists($f_exper_id, $this->instr_names)) {
            $this->instr_names[$f_exper_id] = $f->experiment()->instrument()->name() ;
        }
        return $this->instr_names[$f_exper_id] ;
    }

    /**
     * Lazy loader for experiment names
     * 
     * @param \RegDB\RegDBFile $f
     * @return string
     */
    private function _exper_name ($f) {
        $f_exper_id = $f->exper_id() ;
        if (!array_key_exists($f_exper_id, $this->exper_names)) {
            $this->exper_names[$f_exper_id] = $f->experiment()->name() ;
        }
        return $this->exper_names[$f_exper_id] ;
    }

    /**
     * Lazy loader for iRODS files
     * 
     * @param \RegDB\RegDBFile $f
     * @return array
     */
    private function _irods_files ($f) {

        $f_exper_id = $f->exper_id() ;
        if (!array_key_exists($f_exper_id, $this->irods_files)) {
            $this->irods_files[$f_exper_id] = array() ;
        }

        $f_run = $f->run() ;
        if (!array_key_exists($f_run, $this->irods_files[$f_exper_id])) {

            $runs = $this->SVC->safe_assign (
                $this->SVC->irodsdb()->runs (
                    $this->_instr_name($f) ,
                    $this->_exper_name($f) ,
                    'xtc' ,
                    $f_run ,
                    $f_run) ,
                "failed to load iRODS files for run {$f_run} of experiment id={$f_exper_id}") ;

            foreach ($runs as $irods) {
                if (!array_key_exists($irods->run, $this->irods_files[$f_exper_id])) {
                    $this->irods_files[$f_exper_id][$irods->run] = array() ;
                }
                foreach ($irods->files as $irods_file) {
                    if (!array_key_exists($irods_file->name, $this->irods_files[$f_exper_id][$irods->run])) {
                        $this->irods_files[$f_exper_id][$irods->run][$irods_file->name] = array() ;
                    }
                    array_push($this->irods_files[$f_exper_id][$irods->run][$irods_file->name], $irods_file) ;
                }
            }

            // If iRODS has no file entries for this run then we would rather
            // put an empty array to prevent further requests related to this run.
            // This little optimization may save us a few more trips to the database
            // for other files of the same run.
            
            if (!array_key_exists($f_run, $this->irods_files[$f_exper_id])) {
                $this->irods_files[$f_exper_id][$f_run] = array() ;
            }
        }
        $f_name = "{$f->base_name()}.xtc" ;
        return array_key_exists($f_name, $this->irods_files[$f_exper_id][$f_run]) ?
            $this->irods_files[$f_exper_id][$f_run][$f_name] :
            array() ;
    }

    /**
     * Compose a entry representing a file in a result set
     *
     * @param type $f
     * @return \stdClass
     */
    private function _file2obj ($f) {

        // Gneneral file information

        $created = $f->open_time() ;
        $closed  = $this->_file_closed_time($f) ;

        $obj = new \stdClass ;
        $obj->instr_id = $this->_instr_id($f) ;
        $obj->exper_id = $f->exper_id() ;
        $obj->run      = $f->run() ;
        $obj->type     = 'xtc' ;
        $obj->name     = $f->base_name() ;
        $obj->size     = $this->_file_size2obj($f) ;
        $obj->created  = $this->_time2obj($created) ;
        $obj->closed   = $this->_time2obj($closed) ;

        // migration stage:
        //
        // NOTES:
        // - The begin_delay is measured since file creation time
        // - The end_delay is measured since a moment when the was was supposed
        //   to be closed (if it's known). Otherwise measure it from the file
        //   creation time.

        $DSS2FFB = $obj->DSS2FFB = new \stdClass ;

        $DSS2FFB->host   = '' ;
        $DSS2FFB->status = 'W' ;    // 'WAIT' to be migrated

        $dss_begin = null ;
        $dss_end   = null ;

        $dss = $f->data_migration_file() ;
        if ($dss) {

            $DSS2FFB->host = $dss->host() ;

            $dss_begin = $dss->start_time() ;
            $dss_end   = $dss->stop_time() ;

            switch ($dss->status()) {
                case 'DONE':
                    $DSS2FFB->status = 'C' ;        // the migration is 'COMPLETE'
                    break ;

                case 'WAIT':
                    if ($dss_begin && !$dss_end) {
                        $DSS2FFB->status = 'P' ;    // the migration is 'IN-PROGRESS'
                        break ;
                    }
                    
                    // Any other combination means that a previous attempt to migrate
                    // the file has failed. So we treat this as the 'WAIT' state
                    //
                    // NOTE: that we're going to reset both timestamps
                    //       bellow (in the 'default' clause) as well.

                default:
                    $dss_begin = null ;
                    $dss_end   = null ;
                    break ;
            }
        }
        $DSS2FFB->begin = $this->_time2obj($dss_begin) ;
        $DSS2FFB->end   = $this->_time2obj($dss_end) ;

        $dss_begin_delay = (is_null($dss_begin) ? $this->now->sec : $dss_begin->sec) - $created->sec ;
        $dss_end_delay   = (is_null($dss_end)   ? $this->now->sec : $dss_end  ->sec) - ($closed ? $closed ->sec : $created->sec) ;

        $DSS2FFB->begin_delay = $dss_begin_delay < 0 ? 0 : $dss_begin_delay ;
        $DSS2FFB->end_delay   = $dss_end_delay   < 0 ? 0 : $dss_end_delay ;

        $DSS2FFB->rate = $this->_rate($obj->size->bytes, $dss_begin, $dss_end) ;

        // migration stage:
        //
        // NOTES:
        // - Both delays are measured from the end of the previous stage (if any.
        //   Otherwise use the start of the previous stage.
        // - And keep using the above explained logic until finding some
        //   meaningful time all the way to the file creation time.

        $FFB2ANA = $obj->FFB2ANA = new \stdClass ;

        $FFB2ANA->host   = '' ;
        $FFB2ANA->status = 'W' ;    // 'WAIT' to be migrated

        $ffb_begin = null ;
        $ffb_end   = null ;

        $ffb = $f->data_migration_file('ana') ;
        if ($ffb) {

            $FFB2ANA->host = $ffb->host() ;

            $ffb_begin = $ffb->start_time() ;
            $ffb_end   = $ffb->stop_time() ;

            switch ($ffb->status()) {
                case 'DONE':
                    $FFB2ANA->status = 'C' ;        // the migration is 'COMPLETE'
                    break ;

                case 'WAIT':
                    if ($ffb_begin && !$ffb_end) {
                        $FFB2ANA->status = 'P' ;    // the migration is 'IN-PROGRESS'
                        break ;
                    }
                    
                    // Any other combination means that a previous attempt to migrate
                    // the file has failed. So we treat this as the 'WAIT' state
                    //
                    // NOTE: that we're going to reset both timestamps
                    //       bellow (in the 'default' clause) as well.

                default:
                    $ffb_begin = null ;
                    $ffb_end   = null ;
                    break ;
            }
        }
        $FFB2ANA->begin = $this->_time2obj($ffb_begin) ;
        $FFB2ANA->end   = $this->_time2obj($ffb_end) ;

        $ffb_prev_stage_end_best_guess = $DSS2FFB->end->sec ;
        if (!$ffb_prev_stage_end_best_guess) {
            $ffb_prev_stage_end_best_guess = $DSS2FFB->begin->sec ;
            if (!$ffb_prev_stage_end_best_guess)
                $ffb_prev_stage_end_best_guess = ($closed ? $closed ->sec : $created->sec) ;
        }
        $ffb_begin_delay = (is_null($ffb_begin) ? $this->now->sec : $ffb_begin->sec) - $ffb_prev_stage_end_best_guess ;
        $ffb_end_delay   = (is_null($ffb_end)   ? $this->now->sec : $ffb_end  ->sec) - $ffb_prev_stage_end_best_guess ;

        $FFB2ANA->begin_delay = $ffb_begin_delay < 0 ? 0 : $ffb_begin_delay ;
        $FFB2ANA->end_delay   = $ffb_end_delay   < 0 ? 0 : $ffb_end_delay ;

        $FFB2ANA->rate = $this->_rate($obj->size->bytes, $ffb_begin, $ffb_end) ;

        // migration stage:
        //
        // NOTES:
        // - Both delays are measured from the end of the previous stage (if any.
        //   Otherwise use the start of the previous stage.
        // - And keep using the above explained logic until finding some
        //   meaningful time all the way to the file creation time.

        $ANA2HPSS = $obj->ANA2HPSS = new \stdClass ;

        $ANA2HPSS->host = '' ;

        $irods_disk_ctime = null ;
        $irods_tape_ctime = null ;

        $irods_files = $this->_irods_files($f) ;
        foreach ($irods_files as $irods) {
            switch ($irods->resource) {
                case 'lustre-resc':
                    $irods_disk_ctime = new LusiTime($irods->ctime) ;
                    break ;
                case 'hpss-resc':
                    $irods_tape_ctime = new LusiTime($irods->ctime) ;
                    break ;
            }
        }

        // If the disk replica is available and it's newer (because the file
        // was restored from tape) then use tape replica tim einstead.
        //
        if (($irods_disk_ctime && $irods_tape_ctime) && $irods_tape_ctime->less($irods_disk_ctime)) {
            $irods_disk_ctime = $irods_tape_ctime ;
        }

        // If no disk replica found then use the tape replica time instead
        //
        if (!$irods_disk_ctime && $irods_tape_ctime) {
            $irods_disk_ctime = $irods_tape_ctime ;
        }

        $ANA2HPSS->begin = $this->_time2obj($irods_disk_ctime) ;
        $ANA2HPSS->end   = $this->_time2obj($irods_tape_ctime) ;

        $ana_prev_stage_end_best_guess = $FFB2ANA->end->sec ;
        if (!$ana_prev_stage_end_best_guess) {
            $ana_prev_stage_end_best_guess = $ffb_prev_stage_end_best_guess ;
        }
        $ana_begin_delay = (is_null($irods_disk_ctime) ? $this->now->sec : $irods_disk_ctime->sec) - $ana_prev_stage_end_best_guess ;
        $ana_end_delay   = (is_null($irods_tape_ctime) ? $this->now->sec : $irods_tape_ctime->sec) - $ana_prev_stage_end_best_guess ;

        $ANA2HPSS->begin_delay = $ana_begin_delay < 0 ? 0 : $ana_begin_delay ;
        $ANA2HPSS->end_delay   = $ana_end_delay   < 0 ? 0 : $ana_end_delay ; ;

        $ANA2HPSS->rate = $this->_rate($obj->size->bytes, $irods_disk_ctime, $irods_tape_ctime) ;

        if ($irods_disk_ctime && $irods_tape_ctime) {
            $ANA2HPSS->status = 'C' ;
        } else if ($irods_disk_ctime && !$irods_tape_ctime) {
            $ANA2HPSS->status = 'P' ;
        } else {
            $ANA2HPSS->status = 'W' ;
        }
        
        // Optional correction for files which bypass the FFB2DSS stage:
        //
        // - if the file is registered in iRODS then reset al counters in
        //   the FFB2DSS section.

        if (!$ffb && $irods_disk_ctime) {
            $FFB2ANA->begin = $this->_time2obj(null) ;
            $FFB2ANA->end   = $this->_time2obj(null) ;
            $FFB2ANA->begin_delay = '' ;
            $FFB2ANA->end_delay   = '' ;
            $FFB2ANA->rate = '' ;
            $FFB2ANA->status = '' ;
        }
        
        return $obj ;
    }

    /**
     * Return an array reepresenting file size
     * 
     * @param \RegDB\RegDBFile $f - file entry reported by the DAQ
     * @return array
     */
    private function _file_size2obj ($f) {

        // TODO: Try to see if the file size info is already in iRODS
        //       using either disk or tae replica
        //
        // SUGGESTION: Try caching the iRODS information for the run witin
        //             the object context first to optimize operations with
        //             the service.
        //             This needs to be done alongside with method $this->_file2obj().

        $bytes = 0 ;

        foreach ($this->_irods_files($f) as $f_irods) {
            $bytes = $f_irods->size ;
            break ;
        }

        $obj = new \stdClass ;
        $obj->bytes = '' ;
        $obj->value = '' ;
        $obj->units = '' ;

        if ($bytes) {

            $value  = $bytes ;
            $format = '%d' ;
            $units  = '' ;

            if      ($bytes < KB) {  }
            else if ($bytes < MB) { $value = $bytes / KB; $format = $bytes < 10 * KB ? '%.1f' : '%d'; $units = 'KB' ; }
            else if ($bytes < GB) { $value = $bytes / MB; $format = $bytes < 10 * MB ? '%.1f' : '%d'; $units = 'MB' ; }
            else                  { $value = $bytes / GB; $format = $bytes < 10 * GB ? '%.1f' : '%d'; $units = 'GB' ; }

            $obj->bytes = $bytes ;
            $obj->value = sprintf($format, $value) ;
            $obj->units = $units ;
        }
        return $obj ;
    }

    /**
     * Calculate the avarage transfer rate (MB/s) if available
     * 
     * @param integer $bytes
     * @param \LusiTime\LusiTime $begin_sec
     * @param \LusiTime\LusiTime $end_sec
     * @return string
     */
    private function _rate ($bytes, $begin, $end) {
        if ($bytes && $begin && $end) {
            $mb = $bytes / MB ;
            $delta = $end->to_float() - $begin->to_float() ;
            if ($delta <= 0.) $delta = 1e-9 ;
            return floor($mb / $delta) ;
        }
        return '' ;
    }

    private function _end_of_run ($exper_id, $runnum) {

        if (!array_key_exists($exper_id, $this->logbook_experiments)) {
            $experiment = $this->SVC->safe_assign (
                $this->SVC->logbook()->find_experiment_by_id($exper_id) ,
                "experiment not found for id={$exper_id}"
            ) ;
            $this->logbook_experiments[$exper_id] = array (
                'exper' => $experiment ,
                'runs'  => array ()
            ) ;
        }
        if (!array_key_exists($runnum, $this->logbook_experiments[$exper_id]['runs'])) {
            $this->logbook_experiments[$exper_id]['runs'][$runnum] = $this->SVC->safe_assign (
                $this->logbook_experiments[$exper_id]['exper']->find_run_by_num($runnum) ,
                "no run {$runnum} found in experiment id={$exper_id}"
            ) ;
        }
        return $this->logbook_experiments[$exper_id]['runs'][$runnum]->end_time() ;
    }

    /**
     * Calculate the file close time if possible
     *
     * The algorithm will try (in the given oreder) these options:
     *
     * 1. the create time of the next chunk (if any) within the same stream, or
     * 2. the end of run time (if any), or
     * 3. null
     *
     * @param \RegDB\RegDBFile $f - file entry object reported by the DAQ
     * @return \LusiTime\LusiTime
     */
    private function _file_closed_time ($f) {

        // Cache file attributes fo rthe sake of optimization

        $f_exper_id = $f->exper_id() ;
        $f_run      = $f->run() ;
        $f_stream   = $f->stream() ;
        $f_chunk    = $f->chunk() ;

        // Option #1

        $itr = $f->experiment()->files_itr($f_run, $f_run) ;
        foreach ($itr as $other) {
            if (($other->stream() == $f_stream) &&
                ($other->chunk()  == $f_chunk + 1)) {
                return $other->open_time() ;
            }
        }

        // Option #2 or #3

        return $this->_end_of_run($f_exper_id, $f_run) ;
    }

    /**
     * 
     * @param \LusiTime\LusiTime $t - timestamp object or null
     * @return array
     */
    private function _time2obj ($t) {
        $obj = new \stdClass ;
//        $obj->iso = $t ? $t->toStringShort() : '' ;
        $obj->day = $t ? $t->toStringDay() : '' ;
        $obj->hms = $t ? $t->toStringHMS() : '' ;
        $obj->sec = $t ? $t->sec : '' ;
        return $obj ;
    }

    /**
     * Return a map of all used experiment ids to their names
     *
     * @return type
     */
    public function experiment_names () {
        return $this->exper_names ;
    }
}


/**
 * The class implementing the iterator interface in a scope
 * of an experiment
 */
class ExperimentScopeFileItr extends FileItr {

    // Parameters of the object

    private $exper_id = null ;
    private $run_range = null ;
    
    // Cached for optimization

    private $experiment = null ;

    /**
     * Constructor
     *
     * @param \DataPortal\ServiceJSON $SVC
     * @param integer $exper_id
     * @param array $run_range
     */
    public function __construct ($SVC, $exper_id, $run_range) {
        parent::__construct ($SVC) ;
        $this->exper_id = $exper_id ;
        $this->$run_range = $run_range ;
    }
    
    /**
     * Return an iterator of files in a scope of an experiment and
     * an optional sub-range of runs.
     *
     * @return \RegDB\RegDBFileItr
     */
    protected function get_itr () {
        
        if (is_null($this->experiment)) {
            $this->experiment = $this->SVC->safe_assign (
                $this->SVC->logbook()->find_experiment_by_id($this->exper_id) ,
                "no experiment found for id={$this->exper_id}"
            ) ;
        }

        $min_run = null ;
        $max_run = null ;

        $first_run = $this->experiment->find_first_run() ;
        if ($first_run) {

            $first_run_num = $first_run->num() ;
            $last_run_num  = $this->experiment->find_last_run()->num() ;

            $min_run = $this->run_range['min'] ;
            $max_run = $this->run_range['max'] ;

            $min_run = max($min_run ? $min_run : 0,             $first_run_num) ;
            $max_run = min($max_run ? $max_run : $last_run_num, $last_run_num) ;
        }
        return $this->SVC->safe_assign (
            $this->experiment->regdb_experiment()->files_itr (
                $min_run ,
                $max_run ,
                true ,      /* reverse_order */
                true        /* order_by_time */
            ) ,
            "failed to set up a file iterator for experiment id={$this->exper_id}"
        ) ;                
    }
}

/**
 * The class implementing the iterator interface in a scope
 * of an instrument (or instruments).
 */
class InstrumentScopeFileItr extends FileItr {

    // Parameters of tht object
    
    private $instr_name = null ;
    private $begin_time = null ;
    private $end_time = null ;

    /**
     * Constructor
     *
     * @param \DataPortal\ServiceJSON $SVC
     * @param string $instr_name
     * @param \LusiTime\LusiTime $begin_time
     * @param \LusiTime\LusiTime $end_time
     */
    public function __construct ($SVC, $instr_name, $begin_time, $end_time) {
        parent::__construct ($SVC) ;
        $this->instr_name = $instr_name ;
        $this->begin_time = $begin_time ;
        $this->end_time = $end_time ;
    }
    
    /**
     * Return an iterator of files in a scope of an experiment and
     * an optional sub-range of runs.
     *
     * @return \RegDB\RegDBFileItr
     */
    protected function get_itr () {
        return $this->SVC->safe_assign (
            $this->SVC->regdb()->files_itr (
                $this->instr_name ,
                $this->begin_time ,
                $this->end_time ,
                true ,  /* reverse_order */
                true    /* order_by_time */
            ) ,
            "failed to set up a file iterator accross instruments"
        ) ;
    }
}


/**
 * The utility class encapsulating queries for the file migration delays
 * in the Data Management system. 
 */
class SysMonFileMigrDelays {

    /**
     * Return an iterator over files in teh specified scope.
     * 
     * A scope of the iterator is specified via the input options
     * object of class \stdClass, which may have either of the following
     * sets of data members:
     * 
     *   (exper_id,   run_range)
     *   (instr_name, begin_time, end_time)
     *
     * @param DataPortal\ServiceJSON $SVC
     * @param \stdClass $opt
     * @return \SysMon\FileItr
     */
    public static function iterator (
        $SVC ,
        $opt) {

        return isset($opt->exper_id) && $opt->exper_id ?
            new ExperimentScopeFileItr($SVC, $opt->exper_id,   $opt->run_range) :
            new InstrumentScopeFileItr($SVC, $opt->instr_name, $opt->begin_time, $opt->end_time) ;
    }
}