<?php
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
        $result = $this->connection->query( 'SELECT * FROM "shift" WHERE exper_id='.$this->attr['id'].$extra_condition );
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
        if( !is_null( $end_time )) {
            if( !$begin_time->less( $end_time ))
                die( "begin time of the new shift should be strictly less than the end one" );
        }

        /* Make sure the shift interval is contained within the one
         * of the experiment.
         */
        if( 0 != $this->in_interval( $begin_time ))
            die( "begin time is out of experiment's limits" );
        if( !is_null($end_time)) {
            if( 0 != $this->in_interval($end_time))
                die( "end time is out of experiment's limits" );
        }

        /* Get the last/current shift (if any). We want to make sure that
         * the new one begins afterward. Also, if the current shift
         * is open-ended then we want to get it closed where the new
         * one begins.
         */
        $last_shift = $this->find_last_shift();
        if( !is_null( $last_shift )) {
            if( !$last_shift->begin_time()->less( $begin_time ))
                die( "begin time of new shift falls into the previous run" );
            if( is_null( $last_shift->end_time())) {
                $last_shift->close( $begin_time )
                    or die( "failed to close the current shift");
            }
        }

        $sql = 'INSERT INTO "shift" VALUES('.$this->attr['id']
            .",".LogBookTime::to64from( $begin_time )
            .",".($end_time==null?'NULL':LogBookTime::to64from( $end_time ))
            .",'".$leader."')";
        $result = $this->connection->query( $sql )
            or die ("failed to create new shift: ".mysql_error());

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
        $result = $this->connection->query( 'SELECT * FROM "run" WHERE exper_id='.$this->attr['id'].$extra_condition );
        $nrows = mysql_numrows( $result );
        for( $i=0; $i<$nrows; $i++ ) {
            array_push(
                $list,
                new LogBookRun(
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

    public function create_run ( $num, $begin_time, $end_time=null ) {

        /* Make sure the interval is valid
         */
        if( !is_null($end_time )) {
            if( !$begin_time->less( $end_time ))
                die( "begin time of the new run should be strictly less than the end one" );
        }

        /* Make sure the run interval is contained within the one
         *       of the experiment.
         */
        if( 0 != $this->in_interval($begin_time))
            die( "begin time is out of experiment's limits" );
        if( !is_null($end_time)) {
            if( 0 != $this->in_interval($end_time))
                die( "end time is out of experiment's limits" );
        }

        /* Get the last/current run (if any). We want to make sure that
         * the new one begins afterward. Also, if the current run
         * is open-ended then we want to get it closed where the new
         * one begins.
         */
        $last_run = $this->find_last_run();
        if( !is_null( $last_run )) {
            if( !$last_run->begin_time()->less( $begin_time ))
                die( "begin time of new run falls into the previous run" );
            if( is_null( $last_run->end_time())) {
                $last_run->close( $begin_time )
                    or die( "failed to close the current run");
            }
        }

        /* Proceed to creating new run in the database.
         */
        $run_num = $num > 0 ? $num : $this->allocate_run($num);
        $sql = 'INSERT INTO "run" VALUES(NULL,'.$run_num
            .",".$this->attr['id']
            .",".LogBookTime::to64from($begin_time)
            .",".($end_time==null?'NULL':LogBookTime::to64from($end_time)).")";
        $result = $this->connection->query( $sql )
            or die ("failed to create new run: ".mysql_error());
        return $this->find_run_by_id('(SELECT LAST_INSERT_ID())');
    }

    private function allocate_run () {

        $result = $this->connection->query(
            'SELECT MAX(num) "num" FROM "run" WHERE exper_id='.
            $this->attr['id'] );

        $nrows = mysql_numrows( $result );
        if( $nrows == 1 ) {
            $attr = mysql_fetch_array( $result, MYSQL_ASSOC );
            if(isset($attr['num'])) return 1 + $attr['num'];
            return 1;
        }
        die( "internal error" );
    }

    /* ==========================
     *   SUMMARY RUN PARAMETERS
     * ==========================
     */
    public function run_params ( $condition='' ) {

        $list = array();
        $extra_condition = $condition == '' ? '' : 'AND '.$condition;
        $result = $this->connection->query( 'SELECT * FROM "run_param" WHERE exper_id='.$this->attr['id'].$extra_condition );
        $nrows = mysql_numrows( $result );
        for( $i=0; $i<$nrows; $i++ ) {
            array_push(
                $list,
                new LogBookRunParam(
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
        $result = $this->connection->query(
            'SELECT * FROM "run_param" WHERE exper_id='.
            $this->attr['id'].$extra_condition );

        $nrows = mysql_numrows( $result );
        if( $nrows == 1 )
            return new LogBookRunParam(
                $this->connection,
                $this,
                mysql_fetch_array( $result, MYSQL_ASSOC ));

        return null;
    }

    public function create_run_param ( $param, $type, $descr ) {

        $sql = "INSERT INTO \"run_param\" VALUES(NULL,'".$param
            ."',".$this->attr['id']
            .",'".$type."','".$descr."')";
        $result = $this->connection->query( $sql )
            or die ("failed to create new run parameter: ".mysql_error());

        return $this->find_run_param_by_id( '(SELECT LAST_INSERT_ID())' );
    }

    /* Close the open-ended experiment
     */
    public function close ( $end_time ) {

        if( !is_null($this->attr['end_time']))
            die( "the experiment is already closed" );

        /* Verify the value of the parameter
         */
        if( is_null( $end_time ))
            die( "end time can't be null");
        if( 0 != $this->in_interval( $end_time ))
            die( "incorrect end time - it falls off the experiment's intervall" );

        /* Check the last run to be sure that it would completelly fall (unless
         * its' open-ended) into the truncated limits of the experiment. Close
         * the run at the sam eend tim eas well.
         */
        $last_run = $this->find_last_run();
        if( !is_null( $last_run )) {
            if( $last_run->begin_time()->greaterOrEqual( $end_time ))
                die( "experiment end time is at or before the begin timer of the last run" );
            if( is_null( $last_run->end_time())) {
                $last_run->close( $end_time )
                    or die( "failed to close the current run");
            } else {
                if( $end_time->less( $last_run->end_time()))
                    die( "experiment end time is before the end of the last run" );
            }
        }

        /* Similar action (as for the last run) must be applied to the last
         * shift (if any).
         */
        $last_shift = $this->find_last_shift();
        if( !is_null( $last_shift )) {
            if( $last_shift->begin_time()->greaterOrEqual( $end_time ))
                die( "experiment end time is at or before the begin timer of the last shift" );
            if( is_null( $last_shift->end_time())) {
                $last_shift->close( $end_time )
                    or die( "failed to close the current shift");
            } else {
                if( $end_time->less( $last_shift->end_time()))
                    die( "experiment end time is before the end of the last shift" );
            }
        }

        /* Make the update
         */
        $end_time_64 = LogBookTime::to64from( $end_time );
        $this->connection->query(
            'UPDATE "experiment" SET end_time='.$end_time_64.
            ' WHERE id='.$this->id())
            or die ("failed to close the experiment: ".mysql_error());

        /* Update the current state of the object
         */
        $this->attr['end_time'] = $end_time_64;
        return true;
    }
}
?>
