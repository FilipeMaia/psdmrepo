<?php
/**
 * Class LogBookExperiment an abstraction for experiments.
 *
 * @author gapon
 */
class LogBookExperiment {

    /* Data members
     */
    private $connection;

    public $attr;

    /* Constructor
     */
    public function __construct ( $connection, $attr ) {
        $this->connection = $connection;
        $this->attr = $attr;
    }

    public function id () {
        return $this->attr['id']; }

    public function begin_time () {
        return LogBookTime::from64( $this->attr['begin_time'] ); }

    public function end_time () {
        if( is_null( $this->attr['end_time'] )) return null;
        return LogBookTime::from64( $this->attr['end_time'] ); }

    public function name () {
        return $this->attr['name']; }

    public function in_interval ( $timestamp ) {
        return LogBookTime::in_interval(
            $timestamp,
            $this->attr['begin_time'],
            $this->attr['end_time'] ); }

    /* ==========
     *   SHIFTS
     * ==========
     */
    public function shifts ($condition='') {

        $list = array();

        $extra_condition = $condition == '' ? '' : 'AND '.$condition;
        $result = $this->connection->query(
            'SELECT * FROM "shift" WHERE exper_id='.$this->attr['id'].$extra_condition );

        $nrows = mysql_numrows( $result );
        for( $i=0; $i<$nrows; $i++ ) {
            array_push(
                $list,
                new LogBookShift(
                    $this->connection,
                    $this,
                    mysql_fetch_array( $result, MYSQL_ASSOC )));
        }
        return $list;
    }

    public function create_shift ( $leader, $begin_time, $end_time=null ) {

        /* Make sure the interval is valid
         */
        if( !is_null( $end_time ) && !$begin_time->less( $end_time ))
            throw new LogBookException(
                __METHOD__,
                "begin time '".$begin_time."' isn't less than end time '".$end_time."'" );

        /* Make sure the shift interval is contained within the one
         * of the experiment.
         */
        if( 0 != $this->in_interval( $begin_time ))
            throw new LogBookException(
                __METHOD__,
                "begin time '".$begin_time."' is out of experiment's limits" );

        if( !is_null( $end_time ) && 0 != $this->in_interval( $end_time ))
            throw new LogBookException(
                __METHOD__,
                "end time '".$end_time."' is out of experiment's limits" );

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

        /* Proceed with the new shift.
         */
        $this->connection->query (
            'INSERT INTO "shift" VALUES('.$this->attr['id']
            .",".LogBookTime::to64from( $begin_time )
            .",".( is_null( $end_time ) ? 'NULL' : LogBookTime::to64from( $end_time ))
            .",'".$leader."')" );

        return $this->find_shift_by_begin_time( $begin_time );
    }

    public function find_shift_by_begin_time ( $begin_time ) {
        return $this->find_shift_by_( "begin_time=".LogBookTime::to64from($begin_time)) ; }

    public function find_shift_at ( $time ) {
        return $this->find_shift_by_( 'begin_time <= '.$time.' AND '.$time.'< end_time') ; }

    public function find_last_shift () {
        return $this->find_shift_by_( 'begin_time=(SELECT MAX(begin_time) FROM "shift")' ) ; }

    private function find_shift_by_ ( $condition=null ) {

        $extra_condition = $condition == null ? '' : ' AND '.$condition;
        $result = $this->connection->query(
            'SELECT * FROM "shift" WHERE exper_id='.
            $this->attr['id'].$extra_condition );

        $nrows = mysql_numrows( $result );
        if( $nrows == 1 )
            return new LogBookShift(
                $this->connection,
                $this,
                mysql_fetch_array( $result, MYSQL_ASSOC ));

        return null;
    }

    /* ========
     *   RUNS
     * ========
     */
    public function runs ( $condition='' ) {

        $list = array();

        $extra_condition = $condition == '' ? '' : ' AND '.$condition;
        $result = $this->connection->query(
            'SELECT * FROM "run" WHERE exper_id='.$this->attr['id'].$extra_condition );

        $nrows = mysql_numrows( $result );
        for( $i=0; $i<$nrows; $i++ ) {
            array_push(
                $list,
                new LogBookRun (
                    $this->connection,
                    $this,
                    mysql_fetch_array( $result, MYSQL_ASSOC )));
        }
        return $list;
    }

    public function find_run_by_id ( $id ) {
        return $this->find_run_by_( 'id='.$id) ; }

    public function find_run_by_num ( $num ) {
        return $this->find_run_by_( "num=".$num) ; }

    public function find_last_run () {
        return $this->find_run_by_(
            'id=(SELECT MAX(id) FROM "run" WHERE exper_id='.
            $this->attr['id'].')' ); }

    private function find_run_by_ ( $condition=null ) {

        $extra_condition = $condition == null ? '' : ' AND '.$condition;
        $result = $this->connection->query(
            'SELECT * FROM "run" WHERE exper_id='.
            $this->attr['id'].$extra_condition );

        $nrows = mysql_numrows( $result );
        if( $nrows == 1 )
            return new LogBookRun(
                $this->connection,
                $this,
                mysql_fetch_array( $result, MYSQL_ASSOC ));

        return null;
    }

    /**
     * Create new run
     *
     * @param int $num
     * @param LogBookTime $begin_time
     * @param LogBookTime $end_time
     *
     * @return LogBookRun - new run object
     */
    public function create_run ( $num, $begin_time, $end_time=null ) {

        /* Verify parameters
         */
        if( !is_null( $end_time ) && !$begin_time->less( $end_time ))
            throw new LogBookException(
                __METHOD__,
                "begin time '".$begin_time."' isn't less than end time '".$end_time."'" );

        if( 0 != $this->in_interval( $begin_time ))
            throw new LogBookException(
                __METHOD__,
                "begin time '".$begin_time."' is out of experiment's interval" );

        if( !is_null( $end_time ) && 0 != $this->in_interval( $end_time ))
            throw new LogBookException(
                __METHOD__,
                "end time '".$end_time."' is out of experiment's interval" );

        /* Make sure the new one begins after the previous/last run ends.
         * If that (last) run is still open-ended then get it closed where
         * the new will begin.
         */
        $last_run = $this->find_last_run();
        if( !is_null( $last_run )) {
            if( !$last_run->begin_time()->less( $begin_time ))
                throw new LogBookException(
                    __METHOD__,
                "begin time '".$begin_time."' falls into the previous run's interval" );

            if( is_null( $last_run->end_time()))
                $last_run->close( $begin_time );
        }

        /* Proceed to creating new run in the database.
         */
        $this->connection->query(
            'INSERT INTO "run" VALUES(NULL,'.( $num > 0 ? $num : $this->allocate_run( $num ))
            .",".$this->attr['id']
            .",".LogBookTime::to64from( $begin_time )
            .",".( is_null( $end_time ) ? 'NULL' : LogBookTime::to64from( $end_time )).")" );

        return $this->find_run_by_id('(SELECT LAST_INSERT_ID())');
    }

    /**
     * Get a number of the next available run which doesn't exist yet
     *
     * @return int - next available run number
     */
    private function allocate_run () {

        $result = $this->connection->query(
            'SELECT MAX(num) "num" FROM "run" WHERE exper_id='.
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
        $result = $this->connection->query (
            'SELECT * FROM "run_param" WHERE exper_id='.$this->attr['id'].$extra_condition );

        $nrows = mysql_numrows( $result );
        for( $i=0; $i<$nrows; $i++ ) {
            array_push(
                $list,
                new LogBookRunParam (
                    $this->connection,
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
        $result = $this->connection->query (
            'SELECT * FROM "run_param" WHERE exper_id='.
            $this->attr['id'].$extra_condition );

        $nrows = mysql_numrows( $result );
        if( $nrows == 1 )
            return new LogBookRunParam (
                $this->connection,
                $this,
                mysql_fetch_array( $result, MYSQL_ASSOC ));

        return null;
    }

    public function create_run_param ( $param, $type, $descr ) {

        $this->connection->query (
            "INSERT INTO run_param VALUES(NULL,'".$param.
            "',".$this->attr['id'].
            ",'".$type.
            "','".$descr."')" );

        return $this->find_run_param_by_id( '(SELECT LAST_INSERT_ID())' );
    }

    /* =====================
     *   FREE-FORM ENTRIES
     * =====================
     */

    /**
     * Get all known entries (headers)
     *
     * @return array(LogBookFFEntry)
     */
    public function entries () {

        $list = array();

        $result = $this->connection->query (
            'SELECT h.exper_id, h.relevance_time, e.* FROM header h, entry e WHERE h.exper_id='.$this->attr['id'].
            ' AND h.id = e.hdr_id AND e.parent_entry_id is NULL' );

        $nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ ) {
            array_push(
                $list,
                new LogBookFFEntry (
                    $this->connection,
                    $this,
                    mysql_fetch_array( $result, MYSQL_ASSOC )));
        }
        return $list;
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

        $result = $this->connection->query (
            'SELECT h.exper_id, h.relevance_time, e.* FROM header h, entry e WHERE h.exper_id='.$this->attr['id'].
            ' AND h.id = e.hdr_id ORDER BY relevance_time DESC LIMIT 1' );

        $nrows = mysql_numrows( $result );
        if( $nrows == 1 )
            return new LogBookFFEntry (
                $this->connection,
                $this,
                mysql_fetch_array( $result, MYSQL_ASSOC ));

        return null;
    }

    public function find_entry_by_ ( $condition=null ) {

        $extra_condition = $condition == null ? '' : ' AND '.$condition;
        $result = $this->connection->query (
            'SELECT h.exper_id, h.relevance_time, e.* FROM header h, entry e WHERE h.exper_id='.$this->attr['id'].
            ' AND h.id = e.hdr_id'.$extra_condition );

        $nrows = mysql_numrows( $result );
        if( $nrows == 1 )
            return new LogBookFFEntry (
                $this->connection,
                $this,
                mysql_fetch_array( $result, MYSQL_ASSOC ));

        return null;
    }

    public function create_entry( $relevance_time, $author, $content_type, $content ) {

        if( 0 != $this->in_interval( $relevance_time ))
            throw new LogBookException(
                __METHOD__,
                "relevance time '".$relevance_time."' falls off experiment's interval" );

        $this->connection->query (
            "INSERT INTO header VALUES(NULL,".$this->attr['id'].
            ",".LogBookTime::to64from( $relevance_time ).")" );

        $this->connection->query (
            "INSERT INTO entry VALUES(NULL,(SELECT LAST_INSERT_ID()),NULL".
            ",".LogBookTime::now()->to64().
            ",'".$author.
            "','".$content.
            "','".$content_type."')" );

        return $this->find_entry_by_ (
            'hdr_id = (SELECT h.id FROM header h, entry e'.
            ' WHERE h.id = e.hdr_id AND e.id = (SELECT LAST_INSERT_ID()))' );
    }

    /**
     * Close the open-ended experiment
     *
     * @param LogBookTime $end_time
     */
    public function close ( $end_time ) {

        if( !is_null( $this->attr['end_time'] ))
            throw new LogBookException(
                __METHOD__,
                "the experiment is already closed" );

        /* Verify the value of the parameter
         */
        if( is_null( $end_time ))
            throw new LogBookException(
                __METHOD__,
                "end time can't be null");

        if( 0 != $this->in_interval( $end_time ))
            throw new LogBookException(
                __METHOD__,
                "end time '".$end_time."' falls off experiment's interval" );

        /* Check the last run to be sure that it would completelly fall (unless
         * its' open-ended) into the truncated limits of the experiment. Close
         * the run at the same end time as well.
         */
        $last_run = $this->find_last_run();
        if( !is_null( $last_run )) {
            if( $last_run->begin_time()->greaterOrEqual( $end_time ))
                throw new LogBookException(
                    __METHOD__,
                    "end time '".$end_time."' is before begin time of last run" );

            if( is_null( $last_run->end_time())) {
                $last_run->close( $end_time );
            } else {
                if( $end_time->less( $last_run->end_time()))
                    throw new LogBookException(
                        __METHOD__,
                        "end time '".$end_time."' is before end time of last run" );
            }
        }

        /* Similar action (as for the last run) must be applied to the last
         * shift (if any).
         */
        $last_shift = $this->find_last_shift();
        if( !is_null( $last_shift )) {
            if( $last_shift->begin_time()->greaterOrEqual( $end_time ))
                throw new LogBookException(
                    __METHOD__,
                    "end time '".$end_time."' is before begin time of last shift" );

            if( is_null( $last_shift->end_time())) {
                $last_shift->close( $end_time );
            } else {
                if( $end_time->less( $last_shift->end_time()))
                    throw new LogBookException(
                        __METHOD__,
                        "end time '".$end_time."' is before end time '".
                        $last_shift->end_time()."' of last shit" );
            }
        }

        /* Make sure the last free-form entry (if any) was created before
         * the requested end time.
         */
        $last_entry = $this->find_last_entry();
        if( !is_null( $last_entry ))
            if( $last_entry->relevance_time()->greaterOrEqual( $end_time ))
                throw new LogBookException(
                    __METHOD__,
                    "end time '".$end_time."' is before relevance time '".
                    $last_entry->relevance_time()."' of last free-form entry" );

        /* Make the update
         */
        $end_time_64 = LogBookTime::to64from( $end_time );
        $this->connection->query(
            'UPDATE "experiment" SET end_time='.$end_time_64.
            ' WHERE id='.$this->id());

        /* Update the current state of the object
         */
        $this->attr['end_time'] = $end_time_64;
    }
}
?>
