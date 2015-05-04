<?php

namespace LogBook ;

require_once 'logbook.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

use LusiTime\LusiTime ;
use LusiTime\LusiInterval ;

/**
 * Class LogBookExperiment an abstraction for experiments.
 *
 * @author gapon
 */
class LogBookExperiment {

    /* Data members
     */
    private $logbook ;
    private $regdb_experiment ;

    public $attr ;

    /* Constructor
     */
    public function __construct ($logbook, $regdb_experiment) {
        $this->logbook = $logbook ;
        $this->regdb_experiment = $regdb_experiment ;
        $this->attr = array (
            'id'         => $this->regdb_experiment->id() ,
            'name'       => $this->regdb_experiment->name() ,
            'begin_time' => $this->regdb_experiment->begin_time()->to64() ,
            'end_time'   => $this->regdb_experiment->end_time()->to64()) ;
    }

    public function logbook          ()      { return $this->logbook ; }
    public function regdb_experiment ()      { return $this->regdb_experiment ; }
    public function id               ()      { return $this->regdb_experiment->id            () ; }
    public function name             ()      { return $this->regdb_experiment->name          () ; }
    public function begin_time       ()      { return $this->regdb_experiment->begin_time    () ; }
    public function end_time         ()      { return $this->regdb_experiment->end_time      () ; }
    public function description      ()      { return $this->regdb_experiment->description   () ; }
    public function instrument       ()      { return $this->regdb_experiment->instrument    () ; }
    public function leader_account   ()      { return $this->regdb_experiment->leader_account() ; }
    public function contact_info     ()      { return $this->regdb_experiment->contact_info  () ; }
    public function POSIX_gid        ()      { return $this->regdb_experiment->POSIX_gid     () ; }
    public function in_interval      ($time) { return $this->regdb_experiment->in_interval   ($time) ; }
    public function is_facility      ()      { return $this->regdb_experiment->is_facility   () ; }

    public function days () {
        $ival = new LusiInterval($this->begin_time(), $this->end_time()) ;
        return $ival->splitIntoDays() ;
    }

    /* ==========
     *   SHIFTS
     * ==========
     */
    public function num_shifts ( $condition='' ) {

        /* TODO: This is very inefficient implementation. Replace it by
         * a direct SQL statement for counting rows instead!.
         */
        return count( $this->shifts( $condition )); }



    public function shifts_in_interval ( $begin=null, $end=null ) {

        $condition = '';
        if( !is_null( $begin )) {
            $condition = ' begin_time >= '.$begin->to64();
        }
        if( !is_null( $end )) {
            if( $condition == '' )
                $condition = ' begin_time < '.$end->to64();
            else
                $condition .= ' AND begin_time < '.$end->to64();
        }
        return $this->shifts( $condition );
    }

    public function shifts ( $condition='' ) {

        $list = array();

        $extra_condition = $condition == '' ? '' : ' AND '.$condition;
        $result = $this->logbook->query (
            "SELECT * FROM {$this->logbook->database}.shift WHERE exper_id=".$this->id().$extra_condition.
            ' ORDER BY begin_time DESC' );

        $nrows = mysql_numrows( $result );
        for( $i=0; $i<$nrows; $i++ ) {
            array_push(
                $list,
                new LogBookShift(
                    $this->logbook,
                    $this,
                    mysql_fetch_array( $result, MYSQL_ASSOC )));
        }
        return $list;
    }

    public function create_shift ( $leader, $crew, $begin_time, $end_time=null ) {

        /* Verify the leader's name
         */
        if( is_null( $leader ))
            throw new LogBookException(
                __METHOD__, "crew leader name is null" );

        $leader = trim( $leader );
        if( strlen( $leader ) == 0 )
            throw new LogBookException(
                __METHOD__, "crew leader name is empty" );

        /* Make sure the interval is valid
         */
        if( !is_null( $end_time ) && !$begin_time->less( $end_time ))
            throw new LogBookException(
                __METHOD__,
                "begin time '".$begin_time."' isn't less than end time '".$end_time."'" );

        /* Process the list of crew members, and add the leader into it
         * if the leader isn't there yet. Also make sure the array has
         * no duplicate names, and the names aren't empty.
         */
        $leader_in_crew = false;
        $shift_crew = array_unique( $crew );
        foreach( $shift_crew as $member ) {

            if( is_null( $member ))
                throw new LogBookException(
                    __METHOD__, "crew member name is null" );

            $member = trim( $member );
            if( strlen( $member ) == 0 )
                throw new LogBookException(
                    __METHOD__, "crew member name is empty" );

            if( $member == $leader )
                $leader_in_crew = true;
        }
        if( !$leader_in_crew )
            array_push( $shift_crew, $leader );

        /* Get the last/current shift (if any). We want to make sure that
         * the new one begins afterward. Also, if the current shift
         * is open-ended then we want to get it closed where the new
         * one begins.
         */
        $last_shift = $this->find_last_shift();
        if( !is_null( $last_shift )) {
            if( !$last_shift->begin_time()->less( $begin_time ))
                throw new LogBookException(
                    __METHOD__,
                    "begin time '".$begin_time."' of new shift falls into the previous shift" );

            if( is_null( $last_shift->end_time()))
                $last_shift->close( $begin_time );
        }

        /* Proceed with the new shift and the shift crew.
         */
        $this->logbook->query (
            "INSERT INTO {$this->logbook->database}.shift VALUES(NULL,".$this->attr['id']
            .",".LusiTime::to64from( $begin_time )
            .",".( is_null( $end_time ) ? 'NULL' : LusiTime::to64from( $end_time ))
            .",'".$leader."')" );

        $new_shift = $this->find_shift_by_( 'id=(SELECT LAST_INSERT_ID())' );
        if( is_null( $new_shift ))
            throw new LogBookException(
                __METHOD__,
                "internal implementation errort" );

        foreach( $shift_crew as $member )
            $this->logbook->query (
                "INSERT INTO {$this->logbook->database}.shift_crew VALUES({$new_shift->id()},'$member')" );

        return $new_shift;
    }

    public function find_shift_by_id ( $id ) {
        return $this->find_shift_by_( "id={$id}" ) ; }

    public function find_shift_by_begin_time ( $begin_time ) {
        return $this->find_shift_by_( "begin_time=".LusiTime::to64from($begin_time)) ; }

    public function find_shift_at ( $time ) {
        return $this->find_shift_by_( 'begin_time <= '.$time.' AND (end_time IS NULL OR '.$time.'< end_time)') ; }

    public function find_last_shift () {
        return $this->find_shift_by_( "begin_time=(SELECT MAX(begin_time) FROM {$this->logbook->database}.shift WHERE exper_id={$this->id()})" ) ; }


    public function find_prev_shift_for( $shift ) {
        $sql = <<<HERE
begin_time=(SELECT MAX(begin_time) FROM {$this->logbook->database}.shift WHERE exper_id={$this->id()} AND begin_time<{$shift->begin_time()->to64()} AND id!={$shift->id()})
HERE;
        return $this->find_shift_by_( $sql );
    }
    public function find_next_shift_for( $shift ) {
        $sql = <<<HERE
begin_time=(SELECT MIN(begin_time) FROM {$this->logbook->database}.shift WHERE exper_id={$this->id()} AND begin_time>{$shift->begin_time()->to64()} AND id!={$shift->id()})
HERE;
        return $this->find_shift_by_( $sql );
    }

    private function find_shift_by_ ( $condition=null ) {

        $extra_condition = is_null( $condition ) ? '' : ' AND '.$condition;
        $result = $this->logbook->query(
            "SELECT * FROM {$this->logbook->database}.shift WHERE exper_id=".
            $this->attr['id'].$extra_condition.' ORDER BY begin_time DESC' );

        $nrows = mysql_numrows( $result );
        if( $nrows == 1 )
            return new LogBookShift(
                $this->logbook,
                $this,
                mysql_fetch_array( $result, MYSQL_ASSOC ));

        return null;
    }

    /* ========
     *   RUNS
     * ========
     */
    public function num_runs ( $condition='' ) {

        /* TODO: This is very inefficient implementation. Replace it by
         * a direct SQL statement for counting rows instead!.
         */
        return count( $this->runs( $condition ));
    }

    /**
     * Find runs which began within the specified interval.
     * 
     * @param type $begin - optional begin time of the interval
     * @param type $end - optional end time of the interval
     * @param type $limit - limit the number of entries returned by the operation
     * @return array of runs
     */
    public function runs_in_interval ( $begin=null, $end=null, $limit=null ) {

        $condition = '';
        if( !is_null( $begin )) {
            $condition = ' begin_time >= '.$begin->to64();
        }
        if( !is_null( $end )) {
            if( $condition == '' )
                $condition = ' begin_time < '.$end->to64();
            else
                $condition .= ' AND begin_time < '.$end->to64();
        }
        $limit_str = is_null( $limit ) ? '' : ' LIMIT '.$limit;
        return $this->runs( $condition, $limit_str );
    }

    /**
     * Find runs which either ended or are still going on began within
     * the specified interval. Note that this will also include runs
     * which are wider than the interval, i.e. started before that interval
     * and lasting beyong that interval.
     * 
     * @param type $begin
     * @param type $end
     * @param type $limit
     * @return type 
     */
    public function runs_intersecting_interval ( $begin=null, $end=null, $limit=null ) {

        $b64 = is_null($begin) ? null : $begin->to64();
        $e64 = is_null($end)   ? null : $end->to64();

        $condition = '';

        if(      $b64 && $e64 ) $condition = " (( begin_time < {$b64} AND end_time IS NULL ) OR ( begin_time >= {$b64} AND begin_time < {$e64})) ";
        else if( $b64         ) $condition = " (( begin_time < {$b64} AND end_time IS NULL ) OR   begin_time >= {$b64} )";
        else if(         $e64 ) $condition = "    begin_time < {$e64} ";

        return $this->runs( $condition, is_null($limit) ? '' : ' LIMIT '.$limit );
    }

    public function runs ( $condition='', $limit='' ) {

        $list = array();

        $extra_condition = $condition == '' ? '' : ' AND '.$condition;
        $sql = "SELECT * FROM {$this->logbook->database}.run WHERE exper_id=".$this->attr['id'].$extra_condition.
               ' ORDER BY num, begin_time DESC'.$limit;
        $result = $this->logbook->query($sql);
        $nrows = mysql_numrows( $result );
        for( $i=0; $i<$nrows; $i++ ) {
            array_push(
                $list,
                new LogBookRun (
                    $this->logbook,
                    $this,
                    mysql_fetch_array( $result, MYSQL_ASSOC )));
        }
        return $list;
    }

    public function find_run_by_id ( $id ) {
        return $this->find_run_by_( 'id='.$id) ; }

    public function find_run_by_num ( $num ) {
        return $this->find_run_by_( "num=".$num) ; }

    public function find_first_run () {
        return $this->find_run_by_(
            "num=(SELECT MIN(num) FROM {$this->logbook->database}.run WHERE exper_id=".
            $this->attr['id'].')' ); }

    public function find_last_run () {
        return $this->find_run_by_(
            "num=(SELECT MAX(num) FROM {$this->logbook->database}.run WHERE exper_id=".
            $this->attr['id'].')' ); }

    public function find_prev_run_for( $run ) {
        $sql = <<<HERE
begin_time=(SELECT MAX(begin_time) FROM {$this->logbook->database}.run WHERE exper_id={$this->id()} AND begin_time<{$run->begin_time()->to64()} AND id!={$run->id()})
HERE;
        return $this->find_run_by_( $sql );
    }
    public function find_next_run_for( $run ) {
        $sql = <<<HERE
begin_time=(SELECT MIN(begin_time) FROM {$this->logbook->database}.run WHERE exper_id={$this->id()} AND begin_time>{$run->begin_time()->to64()} AND id!={$run->id()})
HERE;
        return $this->find_run_by_( $sql );
    }

    private function find_run_by_ ( $condition=null ) {

        $extra_condition = $condition == null ? '' : ' AND '.$condition;
        $result = $this->logbook->query(
            "SELECT * FROM {$this->logbook->database}.run WHERE exper_id=".
            $this->attr['id'].$extra_condition.
            ' ORDER BY begin_time DESC' );

        $nrows = mysql_numrows( $result );
        if( $nrows == 1 )
            return new LogBookRun(
                $this->logbook,
                $this,
                mysql_fetch_array( $result, MYSQL_ASSOC ));

        return null;
    }

    /**
     * Create new run
     *
     * @param int $num
     * @param LusiTime $begin_time
     * @param LusiTime $end_time
     *
     * @return LogBookRun - new run object
     */
    public function create_run ( $num, $type, $begin_time, $end_time=null ) {

        /* Verify parameters
         */
        if( is_null( $begin_time ))
            throw new LogBookException(
                __METHOD__,
                "begin time can't be null" );

        if( !is_null( $end_time ) && !$begin_time->less( $end_time ))
            throw new LogBookException(
                __METHOD__,
                "begin time '".$begin_time."' isn't less than end time '".$end_time."'" );

        /* Make sure the new one begins after the previous/last run ends.
         */
        $last_run = $this->find_last_run();
        if( !is_null( $last_run )) {
            if( !$last_run->begin_time()->less( $begin_time ))
                throw new LogBookException(
                    __METHOD__,
                "begin time '".$begin_time."' falls into the previous run's interval" );
        }

        /* Proceed to creating new run in the database.
         */
        $this->logbook->query(
            "INSERT INTO {$this->logbook->database}.run VALUES(NULL,".( $num > 0 ? $num : $this->allocate_run( $num ))
            .",".$this->attr['id']
            .",'".$this->logbook->escape_string( $type )
            ."',".LusiTime::to64from( $begin_time )
            .",".( is_null( $end_time ) ? 'NULL' : LusiTime::to64from( $end_time )).")" );

        return $this->find_run_by_id('(SELECT LAST_INSERT_ID())');
    }

    /**
     * Get a number of the next available run which doesn't exist yet
     *
     * TODO: This operation has to be replaced with a request to
     * the Registration database's Run Numbers generator.
     * 
     * @return int - next available run number
     */
    private function allocate_run () {

        $result = $this->logbook->query(
            "SELECT MAX(num) AS \"num\" FROM {$this->logbook->database}.run WHERE exper_id=".
            $this->attr['id'] );

        $nrows = mysql_numrows( $result );
        if( $nrows == 1 ) {
            $attr = mysql_fetch_array( $result, MYSQL_ASSOC );
            if( isset( $attr['num'] )) return 1 + $attr['num'];
            return 1;
        }
        throw new LogBookException(
            __METHOD__,
            "internal error" );
    }

    /* ==========================
     *   SUMMARY RUN PARAMETERS
     * ==========================
     */
    public function run_params ( $condition='' ) {

        $list = array();

        $extra_condition = $condition == '' ? '' : 'AND '.$condition;
        $result = $this->logbook->query (
            "SELECT * FROM {$this->logbook->database}.run_param WHERE exper_id=".$this->attr['id'].$extra_condition );

        $nrows = mysql_numrows( $result );
        for( $i=0; $i<$nrows; $i++ ) {
            array_push(
                $list,
                new LogBookRunParam (
                    $this->logbook,
                    $this,
                    mysql_fetch_array( $result, MYSQL_ASSOC )));
        }
        return $list;
    }

    public function find_run_param_by_id ( $id ) {
        return $this->find_run_param_by_( 'id='.$id) ; }

    public function find_run_param_by_name ( $name ) {
        return $this->find_run_param_by_ ( "param='".$name."'") ; }

    private function find_run_param_by_ ( $condition=null ) {

        $extra_condition = $condition == null ? '' : ' AND '.$condition;
        $result = $this->logbook->query (
            "SELECT * FROM {$this->logbook->database}.run_param WHERE exper_id=".
            $this->attr['id'].$extra_condition );

        $nrows = mysql_numrows( $result );
        if( $nrows == 1 )
            return new LogBookRunParam (
                $this->logbook,
                $this,
                mysql_fetch_array( $result, MYSQL_ASSOC ));

        return null;
    }

    public function create_run_param ( $param, $type, $descr ) {

        $this->logbook->query (
            "INSERT INTO {$this->logbook->database}.run_param VALUES(NULL,'".$param.
            "',".$this->attr['id'].
            ",'".$type.
            "','".$descr."')" );

        return $this->find_run_param_by_id( '(SELECT LAST_INSERT_ID())' );
    }

    /* =====================
     *   FREE-FORM ENTRIES
     * =====================
     */
    public function num_entries ( $all = false) {

        $result = $this->logbook->query (
            "SELECT COUNT(*) AS 'num_entries' FROM {$this->logbook->database}.header h, {$this->logbook->database}.entry e WHERE h.exper_id=".$this->attr['id'].
            ' AND h.id = e.hdr_id'.
            ($all ? '' : ' AND e.parent_entry_id IS NULL'));

        $nrows = mysql_numrows( $result );
        if( $nrows == 1 ) {
            $attr = mysql_fetch_array( $result, MYSQL_ASSOC );
            return intval($attr['num_entries']);
        }
        return 0;
     }

    /**
     * Get all known entries
     *
     * @return array(LogBookFFEntry)
     */
    public function entries ( $inject_deleted_messages=false ) {
        return $this->entries_by_(
        	$this->sql_4_entries_by_(
        		/* $extra_condition=*/ "",
        		/* $limit=          */ null,
        	    /* $use_tags=       */ false,
        		                       $inject_deleted_messages ));
    }

    /**
     * Get a subset of entries which aren't associated with a shift or a run
     *
     * Note, that this operation would select entroies which aren't
     * explicitly associated with a particular shift or a run.
     *
     * @param LusiTime $begin - the begin time of an interval the entries were posted
     * @param LusiTime $end - the end time of an interval the entries were posted
     *
     * @return array(LogBookFFEntry)
     */
    public function entries_of_experiment ( $begin=null, $end=null, $inject_deleted_messages=false ) {
        if( !is_null( $begin ) && !is_null( $end ) && !$begin->less( $end ))
            throw new LogBookException(
                __METHOD__,
                "begin time '".$begin."' isn't less than end time '".$end."'" );

        return $this->entries_by_(
            $this->sql_4_entries_by_(
                " AND h.shift_id IS NULL\n".
                " AND h.run_id IS NULL\n".
                ( is_null( $begin ) ? "" : " AND e.insert_time >= ".$begin->to64()."\n" ).
                ( is_null( $end )   ? "" : " AND e.insert_time < ".$end->to64()."\n" ),
        		null,  /* $limit= */
        		false, /* $use_tags= */
                $inject_deleted_messages
            )
        );
    }

    /**
     * Get a subset of entries which are associated with the specified shift
     *
     * Note, that this operation would select entroies which are
     * explicitly associated with a particular shift.
     *
     * @return array(LogBookFFEntry)
     */
    public function entries_of_shift ( $id, $inject_deleted_messages=false ) {
        return $this->entries_by_(
        	$this->sql_4_entries_by_(
        		" AND h.shift_id = ".$id."\n",
        		null,  /* $limit= */
        		false, /* $use_tags= */
        		$inject_deleted_messages ));
    }

    /**
     * Get a subset of entries which are associated with the specified run
     *
     * Note, that this operation would select entroies which are
     * explicitly associated with a particular run.
     *
     * @return array(LogBookFFEntry)
     */
    public function entries_of_run ( $id, $inject_deleted_messages=false ) {
        return $this->entries_by_(
        	$this->sql_4_entries_by_(
        		" AND h.run_id = ".$id."\n",
        		null,  /* $limit= */
        		false, /* $use_tags= */
        		$inject_deleted_messages
        	)
        );
    }


    /**
     * The complex search operation
     *
     * The method will search for free-form entries using the specified
     * filter.
     *
     * @return array(LogBookFFEntry)
     */
    public function search (
        $shift_id=null,
        $run_id=null,
        $text2search='',
        $search_in_messages=false,
        $search_in_tags=false,
        $search_in_values=false,
        $posted_at_experiment=false,
        $posted_at_shifts=false,
        $posted_at_runs=false,
        $begin=null,
        $end=null,
        $tag='',
        $author='',
        $since=null,
        $limit=null,
        $inject_deleted_messages=false,
        $search_in_children=false ) {

        /* Verify parameters
         */
        if( !is_null( $shift_id ) && !is_null( $run_id ))
            throw new LogBookException(
                __METHOD__,
                "conflicting parameters: shift_id=".$shift_id." and run_id=".$run_id );

        /* Ignore parameter 'since' if it doesn't fall into an interval of the requst.
         */
        if( !is_null( $since )) {
            if( !is_null( $begin ) && $since->less( $begin )) {
                $since = null;
            }
            if( !is_null( $end ) && $since->greaterOrEqual( $end )) {
                $since = null;
            }
        }

        /* The scope determines at which group of messages to look for:
         *   - those directly attached to the experiment
         *   - attached to shifts
         *   - attached to runs
         * The scope can't empty. Otherwise we'll just return an empty array.
         */
        $scope = '';
        if( $posted_at_experiment && $posted_at_shifts && $posted_at_runs ) {

            /* no fancy subselectors are needed for messages posted in all
             * scopes.
             */

            ;

        } else {
            if( $posted_at_experiment ) {
                $scope = " AND ((h.shift_id IS NULL AND h.run_id IS NULL)";
            }
            if( $posted_at_shifts ) {
                
                // If a specific shift is requested then look for messages which were:
                //
                //   1. either dircetly attached to the shift, or
                //   2. were posted within the shift's boundaries and were not explicitly
                //      associated with another shift
                
                $shift_id_opt = '' ;
                if ($shift_id) {
                    $shift = $this->find_shift_by_id( $shift_id );
                    if( is_null( $shift ))
                        throw new LogBookException(
                            __METHOD__,
                            "no shift with shift_id=".$shift_id." found" );
                    $shift_begin = $shift->begin_time();
                    $shift_end   = $shift->end_time();
                    $shift_begin_opt = is_null( $shift_begin ) ? "" : " AND e.insert_time >= {$shift_begin->to64()}";
                    $shift_end_opt   = is_null( $shift_end   ) ? "" : " AND e.insert_time <  {$shift_end->to64()}";
                    $shift_id_opt = "((h.shift_id={$shift_id}) OR (h.shift_id IS NULL {$shift_begin_opt} {$shift_end_opt}))" ;
                } else {
                    $shift_id_opt = '(h.shift_id IS NOT NULL)';
                }
                if( $scope == "" )
                    $scope = " AND ({$shift_id_opt}";
                else
                    $scope .= " OR {$shift_id_opt}";
            }
            if( $posted_at_runs ) {

                // If a specific run is requested then look for messages which were:
                //
                //   1. either dircetly attached to the run, or
                //   2. were posted within the run's boundaries and were not explicitly
                //      associated with another run

                $run_id_opt = '' ;
                if ($run_id) {
                    $run = $this->find_run_by_id( $run_id );
                    if( is_null( $run ))
                        throw new LogBookException(
                            __METHOD__,
                            "no run with run_id=".$run_id." found" );
                    $run_begin = $run->begin_time();
                    $run_end   = $run->end_time();
                    $run_begin_opt = is_null( $run_begin ) ? "" : " AND e.insert_time >= {$run_begin->to64()}";
                    $run_end_opt   = is_null( $run_end   ) ? "" : " AND e.insert_time <  {$run_end->to64()}";
                    $run_id_opt = "((h.run_id={$run_id}) OR (h.run_id IS NULL {$run_begin_opt} {$run_end_opt}))" ;
                } else {
                    $run_id_opt = '(h.run_id IS NOT NULL)';
                }
                if( $scope == "")
                    $scope = " AND ({$run_id_opt}";
                else
                    $scope .= " OR {$run_id_opt}";
            }
            if( $scope != "" )
                $scope .= ")\n";
            else
                return Array();
        }

        /* Decide at which part(a) of messages to look for. This is actually
         * an optional filter.
         */
        $part = "";
        if( $text2search != "" ) {
            if( $search_in_messages ) {
                $part = " AND ((e.content COLLATE latin1_swedish_ci LIKE '%{$text2search}%')";
            }
            if( $search_in_tags ) {
                if( $part == "" )
                    $part = " AND ((t.tag COLLATE latin1_swedish_ci LIKE '%{$text2search}%' AND t.hdr_id = h.id)";
                else
                    $part .= " OR (t.tag COLLATE latin1_swedish_ci LIKE '%{$text2search}%' AND t.hdr_id = h.id)";
            }
            if( $search_in_values ) {
                if( $part == "" )
                    $part = " AND ((t.value COLLATE latin1_swedish_ci LIKE '%{$text2search}%' AND t.hdr_id = h.id)";
                else
                    $part .= " OR (t.value COLLATE latin1_swedish_ci LIKE '%{$text2search}%' AND t.hdr_id = h.id)";
            }
        }
        if( $part != "" )
            $part .= ")\n";

        /* Consider timing constrains as well (if present).
         */
        if( !is_null( $begin ) && !is_null( $end ) && !$begin->less( $end ))
            throw new LogBookException(
                __METHOD__,
                "begin time '".$begin."' isn't less than end time '".$end."'" );

        $begin_str = is_null( $begin ) ? "" : " AND e.insert_time >= ".$begin->to64()."\n";
        $end_str   = is_null( $end   ) ? "" : " AND e.insert_time < ".$end->to64()."\n";
        $since_str = is_null( $since ) ? "" : " AND e.insert_time > ".$since->to64()."\n";

        /* Consider tag and/or author constrains as well (if present).
         */
        $tag_str    = $tag    == "" ? "" : " AND t.tag = '{$tag}' AND t.hdr_id = h.id\n";
        $author_str = $author == "" ? "" : " AND e.author = '{$author}'\n";

        /* The flag which would instruct the query processor to use the tags table.
         */
        $use_tags = (( $text2search != "" ) && $search_in_tags ) || ( $tag != "" );

        /* Build the extra condition and proceed with the actual search. Expect results
         * to be ordered by the insert (post) time.
         *
         * TODO: Modify this to produce UNION of SELECTs if 'part' includes search
         * in tags (either names or values) and in message bodies. Replace the above
         * presented generator for 'part' accordingly.
         */
        $condition = "";
        $condition .= $scope;
        $condition .= $part;
        $condition .= $begin_str;
        $condition .= $end_str;
        $condition .= $since_str;
        $condition .= $tag_str;
        $condition .= $author_str;

//        if ($shift_id == 927)
//            file_put_contents('php://stderr', print_r("\n".$this->sql_4_entries_by_(
//                $condition,
//                $limit,
//                $use_tags,
//                $inject_deleted_messages,
//                $search_in_children
//            ), TRUE));

        return $this->entries_by_(
            $this->sql_4_entries_by_(
                $condition,
                $limit,
                $use_tags,
                $inject_deleted_messages,
                $search_in_children
            )
        );
    }

    private function sql_4_entries_by_ ( $extra_condition="", $limit=null, $use_tags=true, $inject_deleted_messages=false, $search_in_children=false ) {

    	//throw new LogBookException(
        //	__METHOD__,
        //    $inject_deleted_messages ? "" : " AND e.deleted_time is NULL\n" );

        /* Apply the limit if specified
         *
         * IMPORTANT: Note an order of elements returned by the query. Elements will start
         * from the newest entry. We need it to make sure the limit woud work. In the end
         * (before returning from the method) the array will be reversed.
         */
        $tables = $this->logbook->database.'.header h, '.
                  $this->logbook->database.'.entry e'.
                  ($use_tags ? ', '.$this->logbook->database.'.tag t ' : ' ');
        return
            "SELECT h.exper_id, h.shift_id, h.run_id, h.relevance_time, e.*\n".
            "FROM $tables\n".
            "WHERE h.exper_id = ".$this->attr['id']."\n".
            " AND h.id = e.hdr_id\n".
            ( $search_in_children ? "" : " AND e.parent_entry_id is NULL\n" ).
        	( $inject_deleted_messages ? "" : " AND e.deleted_time is NULL\n" ).
            $extra_condition.
            "ORDER BY e.insert_time DESC\n".
            ( is_null( $limit ) ? "" : "LIMIT $limit\n" );
    }

    private function entries_by_ ( $sql ) {

    	/* DEBUG:
		throw new LogBookException(
        	__METHOD__,
            $sql );
*/
    	$list = array();
/*
$debug_file = fopen( "/tmp/search.txt", "a+" );
fwrite( $debug_file, $sql."\n" );
fclose( $debug_file );
*/
        /* DEBUG: uncomment this exception to see the SQL statement in the client
         * session rather than a result of actual execution.

        throw new LogBookException(
            __METHOD__, $sql );
        */

        $result = $this->logbook->query ( $sql );
        $nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ ) {
            array_push(
                $list,
                new LogBookFFEntry (
                    $this->logbook,
                    $this,
                    mysql_fetch_array( $result, MYSQL_ASSOC )));
        }

        /* IMPORTANT: This is needed because the quesry returned elements
         * in the wrong order (DESC instead of ASC). We had to do it to apply
         * an optinal 'LIMIT';
         */ 
        return array_reverse( $list );
    }

    /**
     * Find the specified entry
     *
     * @param int $id
     * @return LogBookFFEntry
     */
    public function find_entry_by_id ( $id ) {
        return $this->find_entry_by_( 'e.id='.$id ) ; }

    /**
     * Find the last entry (header)
     *
     * @return LogBookFFEntry or null
     */
    public function find_last_entry () {

        $result = $this->logbook->query (
            "SELECT h.exper_id, h.shift_id, h.run_id, h.relevance_time, e.* FROM {$this->logbook->database}.header h, {$this->logbook->database}.entry e WHERE h.exper_id=".$this->attr['id'].
            ' AND h.id = e.hdr_id ORDER BY relevance_time DESC LIMIT 1' );

        $nrows = mysql_numrows( $result );
        if( $nrows == 1 )
            return new LogBookFFEntry (
                $this->logbook,
                $this,
                mysql_fetch_array( $result, MYSQL_ASSOC ));

        return null;
    }

    public function find_entry_by_ ( $condition=null ) {

        $extra_condition = $condition == null ? '' : ' AND '.$condition;
        $result = $this->logbook->query (
            "SELECT h.exper_id, h.shift_id, h.run_id, h.relevance_time, e.* FROM {$this->logbook->database}.header h, {$this->logbook->database}.entry e WHERE h.exper_id=".$this->attr['id'].
            ' AND h.id = e.hdr_id'.$extra_condition );

        $nrows = mysql_numrows( $result );
        if( $nrows == 1 )
            return new LogBookFFEntry (
                $this->logbook,
                $this,
                mysql_fetch_array( $result, MYSQL_ASSOC ));

        return null;
    }

    public function create_entry (
        $author, $content_type, $content,
        $shift_id=null,
        $run_id=null,
        $relevance_time=null ) {

        $insert_time = is_null( $relevance_time ) ? LusiTime::now() : $relevance_time;

        $this->logbook->query (
            "INSERT INTO {$this->logbook->database}.header VALUES(NULL,".$this->id().
            ",".( is_null( $shift_id       ) ? 'NULL' : $shift_id ).
            ",".( is_null( $run_id         ) ? 'NULL' : $run_id ).
            ",".LusiTime::to64from( $insert_time ).")" );

        $this->logbook->query (
            "INSERT INTO {$this->logbook->database}.entry VALUES(NULL,(SELECT LAST_INSERT_ID()),NULL".
            ",".$insert_time->to64().
            ",'".$author.
            "','".$this->logbook->escape_string( $content ).
            "','".$content_type."',NULL,NULL)" );

        return $this->find_entry_by_ (
            "hdr_id = (SELECT h.id FROM {$this->logbook->database}.header h, {$this->logbook->database}.entry e".
            ' WHERE h.id = e.hdr_id AND e.id = (SELECT LAST_INSERT_ID()))' );
    }

    public function delete_entry ( $id, $deleted_time, $deleted_by ) {
        $this->logbook->query (
            "UPDATE {$this->logbook->database}.entry SET deleted_time=".$deleted_time->to64().
            ", deleted_by='".$this->logbook->escape_string( trim( $deleted_by )).
            "' WHERE id=".$id );
    }

    public function undelete_entry ( $id ) {
        $this->logbook->query (
            "UPDATE {$this->logbook->database}.entry SET deleted_time=NULL, deleted_by=NULL WHERE id=".$id );
    }

    /* =========================
     *   PERSISTENT RUN TABLES
     * =========================
     */
    public function run_tables () { return $this->find_run_tables_by_() ; }

    public function create_run_table ($name, $descr, $uid, $coldef) {

        $now = LusiTime::now()->to64() ;
        $uid = $this->logbook->escape_string(trim($uid)) ;
        $this->logbook->query (
            "INSERT INTO {$this->logbook->database}.run_table VALUES(NULL,".$this->id()
            .",'".$this->logbook->escape_string(trim($name))
            ."','".$this->logbook->escape_string(trim($descr))
            ."','".$uid
            ."',".$now
            .",'".$uid
            ."',".$now
            .")"
        ) ;
        $tables = $this->find_run_tables_by_('id IN (SELECT LAST_INSERT_ID())') ;
        if (count($tables) !== 1)
            throw new LogBookException (
                __METHOD__ ,
                "internal error") ;

        $table = $tables[0] ;

        $is_editable = false ;
        switch ($col->type) {
            case 'Editable':
                $is_editable = true ;
                break ;
            case 'Run Info':
                if ($col->source === 'Run Title') $is_editable = true ;
                break ;
        }
        
        foreach ($coldef as $col)
            $table->add_column (
                $col->name ,
                $col->type ,
                $col->source ,
                $col->type === 'Editable',
                $col->position) ;

        return $table ;
    }
    public function find_run_table_by_id ($id) {
        if (!is_numeric($id))
            throw new LogBookException (
                __METHOD__ ,
                "illegal table identifier. A positive number was expected.") ;

        $tables = $this->find_run_tables_by_("id={$id}") ;
        $num_tables = count($tables) ;
        if (!$num_tables) return null ;
        if ($num_tables === 1) return $tables[0] ;
        throw new LogBookException (
            __METHOD__ ,
            "internal error") ;
    }
    private function find_run_tables_by_ ($condition='') {
        $extra_condition = $condition === '' ? '' : ' AND '.$condition ;
        $sql = "SELECT * FROM {$this->logbook->database}.run_table WHERE exper_id={$this->id()}"
            .$extra_condition
            ." ORDER BY name" ;
        $result = $this->logbook->query($sql) ;
        $list = array() ;
        for ($nrows=mysql_numrows($result), $i=0; $i<$nrows; $i++)
            array_push (
                $list ,
                new LogBookRunTable (
                    $this ,
                    mysql_fetch_array($result, MYSQL_ASSOC))) ;

        return $list ;
    }

    public function delete_run_table_by_id ($id) {
        if (!is_numeric($id))
            throw new LogBookException (
                __METHOD__ ,
                "illegal table identifier. A positive number was expected.") ;

        $sql = "DELETE FROM {$this->logbook->database}.run_table WHERE exper_id={$this->id()} AND id={$id}";
        $this->logbook->query($sql) ;
    }
    
    /* ====================
     *   OTHER OPERATIONS
     * ====================
     */
    public function used_tags () {

        $list = array();

        $result = $this->logbook->query (
            "SELECT DISTINCT t.tag FROM {$this->logbook->database}.tag t, {$this->logbook->database}.header h WHERE h.id=t.hdr_id AND h.exper_id=".$this->id().
            ' ORDER BY tag' );

        $nrows = mysql_numrows( $result );
        for( $i=0; $i<$nrows; $i++ )
            array_push( $list,mysql_result( $result, $i ));

        return $list;
    }

    public function used_authors () {

        $list = array();

        $result = $this->logbook->query (
            "SELECT DISTINCT e.author FROM {$this->logbook->database}.entry e, {$this->logbook->database}.header h WHERE h.id=e.hdr_id AND h.exper_id=".$this->id().
            ' ORDER BY author' );

        $nrows = mysql_numrows( $result );
        for( $i=0; $i<$nrows; $i++ )
            array_push( $list,mysql_result( $result, $i ));

        return $list;
    }

    /* =========================================
     *   SUBSCRPTIONS TO RECEIVE NOTIFICATIONS
     * =========================================
     */
    public function subscribe( $subscriber, $address, $time, $host ) {

    	$time64 = LusiTime::to64from( $time );
    	$this->logbook->query (
            "INSERT INTO {$this->logbook->database}.subscriber VALUES(NULL,{$this->id()},'{$subscriber}','{$address}',{$time64},'{$host}')" );

    	$s = $this->find_subscriber_by_( "s.id=(SELECT LAST_INSERT_ID())");
    	if( is_null( $s ))
            throw new LogBookException(
                __METHOD__,
                "internal error" );

        $url     = "https://".$_SERVER['SERVER_NAME'].'/apps/portal/?exper_id='.$this->id()."&app=elog:subscribe";
        $logbook = $this->instrument()->name().'/'.$this->name();

        $body =<<<HERE
URL:     {$url}
LogBook: {$logbook}

** ATTENTION **
The message was sent by the automated notification system because this e-mail
has been just registered to recieve updates on new entries posted in the Electornic LogBook
of the experiment. The registration was made by user '{$s->subscriber()}'
on {$s->subscribed_time()->toStringShort()} from {$s->subscribed_host()}. To unsubscribe
from this service, please follow the above shown URL, select the specified experiment and
proceed to the 'Subscribe' at the top top-level menu of the LogBook application. In case
if you won't be able to log onto the LogBook get in touch with the experiment management
personnel.
HERE;
                
        $this->do_notify( $s->address(), $logbook, "*** SUBSCRIBED ***", $body );
        
        return $s;
    }

    public function unsubscribe( $id ) {

    	$s = $this->find_subscriber_by_( "s.id={$id}");
    	if( is_null( $s ))
            throw new LogBookException(
                __METHOD__,
                "no subscriber for id={$id} error" );

    	$this->logbook->query (
            "DELETE FROM {$this->logbook->database}.subscriber WHERE id={$id}" );

        $url     = "https://".$_SERVER['SERVER_NAME'].'/apps/portal/?exper_id='.$this->id()."&app=elog:subscribe";
        $logbook = $this->instrument()->name().'/'.$this->name();

        $body =<<<HERE
URL:     {$url}
LogBook: {$logbook}

** ATTENTION **
The message was sent by the automated notification system because this e-mail 
has been removed from a list recipients who receive updates on new entries posted 
in the Electornic LogBook of the experiment. The prior registration was made by 
user '{$s->subscriber()}' on {$s->subscribed_time()->toStringShort()} from {$s->subscribed_host()}. 
To subscribe to this service again, please follow the above
shown URL, then select the specified experiment and proceed to the 'Subscribe' at 
the top top-level menu of the LogBook application. In case if you won't be able 
to log onto the LogBook get in touch with the experiment management personnel.
HERE;
                
        $this->do_notify( $s->address(), $logbook, "*** UNSUBSCRIBED ***", $body );
    }

    public function find_subscriber_by_( $condition ) {
        $result = $this->logbook->query (
            "SELECT * FROM {$this->logbook->database}.subscriber s WHERE s.exper_id={$this->id()} AND {$condition}" );

        $nrows = mysql_numrows( $result );
        if( $nrows == 0 ) return null;
        if( $nrows == 1 )
            return new LogBookSubscription (
                $this->logbook,
                $this,
                mysql_fetch_array( $result, MYSQL_ASSOC ));

        throw new LogBookException(
            __METHOD__,
            "inconsistent results returned by the query" );
    }
    public function find_subscriber_by_id( $id ) {
    	return $this->find_subscriber_by_( "s.id={$id}");
    }

    public function subscriptions( $subscribed_by=null ) {

    	$list = array();

        // Find all subscribers. If none then just quit.
        //
        // TODO: Select authorized subscribers only.
        //
        $extra = is_null( $subscribed_by ) ? "" : "AND s.subscriber='{$subscribed_by}'";
        $result = $this->logbook->query (
            "SELECT * FROM {$this->logbook->database}.subscriber s WHERE s.exper_id={$this->id()} {$extra} ORDER BY s.subscriber" );

        $nrows = mysql_numrows( $result );
        for( $i=0; $i<$nrows; $i++ )
            array_push(
                $list,
                new LogBookSubscription (
                    $this->logbook,
                    $this,
                    mysql_fetch_array( $result, MYSQL_ASSOC )));
        return $list;
    }

    /**
     * 
     * @param {LogBookFFEntry} $entry
     * @param {bool} $new_vs_modified
     * @return undefined
     */
    public function notify_subscribers ($entry, $new_vs_modified=true) {

        $subscriptions = $this->subscriptions() ;
        if (count($subscriptions) <= 0) return ;

        $url     = ($_SERVER[HTTPS] ? "https://" : "http://" ).$_SERVER['HTTP_HOST'].'/apps/logbook/index.php?action=select_message&id='.$entry->id() ;
        $author  = $entry->author() ;
        $time    = $entry->insert_time()->toStringShort() ;
        $logbook = $this->instrument()->name().'/'.$this->name() ;
        $subject = strtr(substr($entry->content(), 0, 72), array("'" => '"', "\n" => " ")) ;
        $time_str = $new_vs_modified ? 'Posted  ' : 'Modified' ;

        $entry_str = '' ;
        foreach (explode("\n", $entry->content()) as $line) {
            $entry_str .= "  {$line}\n" ;
        }

        $attachments_base_url = ($_SERVER[HTTPS] ? "https://" : "http://" ).$_SERVER['HTTP_HOST'].'/apps/logbook/attachments' ;
        $attachments_str = '' ;
        foreach ($entry->attachments() as $a) {
            $attachments_str .= "  {$attachments_base_url}/{$a->id()}/{$a->description()}\n" ;
        }
        
        $tags_str = '  ' ;
        foreach ($entry->tags() as $t) {
            $tags_str .= "{$t->tag()} " ;
        }

        foreach ($subscriptions as $s) {
           
            $body =<<<HERE
_______
SUMMARY

  Message:    {$url}
  Author:     {$author}
  {$time_str}:   {$time}
  Experiment: {$logbook}
____________
MESSAGE TEXT

{$entry_str}
___________
ATTACHMENTS

{$attachments_str}
____
TAGS

{$tags_str}
______________________
YOUR SUBSCRIPTION INFO

  The message was sent by the automated  notification system because your
  SLAC e-mail address was registered to  recieve updates on  new  content
  posted in the Electornic LogBook of  the  experiment. The  registration
  was made by user '{$s->subscriber()}' on {$s->subscribed_time()->toStringShort()} from host {$s->subscribed_host()}.
  To unsubscribe from this service, please follow the above shown URL and
  Unsubscribe yourself at the 'Subscribe' menu of the LogBook application.
  In case if you won't be able to log onto the LogBook get in touch  with
  the experiment management personnel.
_________________________________________________________________________
HERE;
            $this->do_notify ($s->address(), $logbook, $subject, $body) ;
        }
    }

    /**
     * 
     * @param {LogBookFFEntry} $entry
     * @param {bool} $new_vs_modified
     * @return undefined
     */
    public function forward ($entry, $recipients, $requestor_uid) {

        if (count($recipients) <= 0) return ;

        $url     = ($_SERVER[HTTPS] ? "https://" : "http://" ).$_SERVER['HTTP_HOST'].'/apps/logbook/index.php?action=select_message&id='.$entry->id() ;
        $author  = $entry->author() ;
        $time    = $entry->insert_time()->toStringShort() ;
        $logbook = $this->instrument()->name().'/'.$this->name() ;
        $subject = strtr(substr($entry->content(), 0, 72), array("'" => '"', "\n" => " ")) ;
        $time_str = 'Posted  ';

        $entry_str = '' ;
        foreach (explode("\n", $entry->content()) as $line) {
            $entry_str .= "  {$line}\n" ;
        }

        $attachments_base_url = ($_SERVER[HTTPS] ? "https://" : "http://" ).$_SERVER['HTTP_HOST'].'/apps/logbook/attachments' ;
        $attachments_str = '' ;
        foreach ($entry->attachments() as $a) {
            $attachments_str .= "  {$attachments_base_url}/{$a->id()}/{$a->description()}\n" ;
        }
        
        $tags_str = '  ' ;
        foreach ($entry->tags() as $t) {
            $tags_str .= "{$t->tag()} " ;
        }

        foreach ($recipients as $r) {
           
            $body =<<<HERE
_______
SUMMARY

  Message:    {$url}
  Author:     {$author}
  {$time_str}:   {$time}
  Experiment: {$logbook}
____________
MESSAGE TEXT

{$entry_str}
___________
ATTACHMENTS

{$attachments_str}
____
TAGS

{$tags_str}
_______________
THE SENDER INFO

  The message was forwarded to you upon a request made by user '{$requestor_uid}'.
  Please, contact that pearson directly if you think the request was made
  by a misstake.
_________________________________________________________________________
HERE;
            $this->do_notify ($r, $logbook, $subject, $body) ;
        }
    }

    private function do_notify( $address, $logbook, $subject, $body ) {
        $tmpfname = tempnam("/tmp", "logbook");
        $handle = fopen( $tmpfname, "w" );
        fwrite( $handle, $body );
        fclose( $handle );

        shell_exec( "cat {$tmpfname} | mail -s '[ {$logbook} ]  {$subject}' {$address} -- -F 'LCLS E-Log'" );

        // Delete the file only after piping its contents to the mailer command.
        // Otherwise its contents will be lost before we use it.
        //
        unlink( $tmpfname );
    }
}
?>
