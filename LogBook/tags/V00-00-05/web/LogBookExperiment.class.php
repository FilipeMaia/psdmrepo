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
    private $regdb_experiment;

    public $attr;

    /* Constructor
     */
    public function __construct ( $connection, $regdb_experiment ) {
        $this->connection = $connection;
        $this->regdb_experiment = $regdb_experiment;
        $this->attr = array (
            'id'         => $this->regdb_experiment->id(),
            'name'       => $this->regdb_experiment->name(),
            'begin_time' => $this->regdb_experiment->begin_time()->to64(),
            'end_time'   => $this->regdb_experiment->end_time()->to64());
    }

    public function id () {
        return $this->regdb_experiment->id(); }

    public function name () {
        return $this->regdb_experiment->name(); }

    public function begin_time () {
        return $this->regdb_experiment->begin_time(); }

    public function end_time () {
        return $this->regdb_experiment->end_time(); }

    public function description () {
        return $this->regdb_experiment->description(); }

    /* ==========
     *   SHIFTS
     * ==========
     */
    public function num_shifts ( $condition='' ) {

        /* TODO: This is very inefficient implementation. Replace it by
         * a direct SQL statement for counting rows instead!.
         */
        return count( $this->shifts( $condition )); }

    public function shifts ( $condition='' ) {

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
        $this->connection->query (
            'INSERT INTO shift VALUES(NULL,'.$this->attr['id']
            .",".LusiTime::to64from( $begin_time )
            .",".( is_null( $end_time ) ? 'NULL' : LusiTime::to64from( $end_time ))
            .",'".$leader."')" );

        $new_shift = $this->find_shift_by_( 'id=(SELECT LAST_INSERT_ID())' );
        if( is_null( $new_shift ))
            throw new LogBookException(
                __METHOD__,
                "internal implementation errort" );

        foreach( $shift_crew as $member )
            $this->connection->query (
                "INSERT INTO shift_crew VALUES({$new_shift->id()},'$member')" );

        return $new_shift;
    }

    public function find_shift_by_begin_time ( $begin_time ) {
        return $this->find_shift_by_( "begin_time=".LusiTime::to64from($begin_time)) ; }

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
    public function num_runs ( $condition='' ) {

        /* TODO: This is very inefficient implementation. Replace it by
         * a direct SQL statement for counting rows instead!.
         */
        return count( $this->runs( $condition )); }

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

    public function find_next_run_for( $prev_run ) {
        $sql = <<<HERE
begin_time=(SELECT MIN(begin_time) FROM "run" WHERE exper_id={$this->id()} AND begin_time>{$prev_run->begin_time()->to64()} AND id!={$prev_run->id()})
HERE;
        return $this->find_run_by_( $sql );
    }

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
     * @param LusiTime $begin_time
     * @param LusiTime $end_time
     *
     * @return LogBookRun - new run object
     */
    public function create_run ( $num, $begin_time, $end_time=null ) {

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
        $this->connection->query(
            'INSERT INTO "run" VALUES(NULL,'.( $num > 0 ? $num : $this->allocate_run( $num ))
            .",".$this->attr['id']
            .",".LusiTime::to64from( $begin_time )
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
    public function num_entries ( $all = false) {

        /* TODO: This is very inefficient implementation. Replace it by
         * a direct SQL statement for counting rows instead!.
         */
        return count( $this->entries()); }

    /**
     * Get all known entries
     *
     * @return array(LogBookFFEntry)
     */
    public function entries () {
        return $this->entries_by_();
    }

    /**
     * Get a subset of entries which aren't associated with a shift or a run
     *
     * Note, that this operation would select entroies which aren't
     * explicitly associated with a particular shift or a run.
     *
     * @return array(LogBookFFEntry)
     */
    public function entries_of_experiment () {
        return $this->entries_by_( 'h.shift_id IS NULL AND h.run_id IS NULL' );
    }

    /**
     * Get a subset of entries which are associated with the specified shift
     *
     * Note, that this operation would select entroies which are
     * explicitly associated with a particular shift.
     *
     * @return array(LogBookFFEntry)
     */
    public function entries_of_shift ( $id ) {
        return $this->entries_by_( 'h.shift_id='.$id );
    }

    /**
     * Get a subset of entries which are associated with the specified run
     *
     * Note, that this operation would select entroies which are
     * explicitly associated with a particular run.
     *
     * @return array(LogBookFFEntry)
     */
    public function entries_of_run ( $id ) {
        return $this->entries_by_( 'h.run_id='.$id );
    }

    private function entries_by_ ( $condition=null ) {

        $list = array();

        $extra_condition = $condition == null ? '' : ' AND '.$condition;
        $result = $this->connection->query (
            'SELECT h.exper_id, h.shift_id, h.run_id, h.relevance_time, e.* FROM header h, entry e WHERE h.exper_id='.$this->attr['id'].
            ' AND h.id = e.hdr_id AND e.parent_entry_id is NULL'.$extra_condition );

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
            'SELECT h.exper_id, h.shift_id, h.run_id, h.relevance_time, e.* FROM header h, entry e WHERE h.exper_id='.$this->attr['id'].
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
            'SELECT h.exper_id, h.shift_id, h.run_id, h.relevance_time, e.* FROM header h, entry e WHERE h.exper_id='.$this->attr['id'].
            ' AND h.id = e.hdr_id'.$extra_condition );

        $nrows = mysql_numrows( $result );
        if( $nrows == 1 )
            return new LogBookFFEntry (
                $this->connection,
                $this,
                mysql_fetch_array( $result, MYSQL_ASSOC ));

        return null;
    }

    public function create_entry (
        $author, $content_type, $content,
        $shift_id=null,
        $run_id=null,
        $relevance_time=null ) {

        $this->connection->query (
            "INSERT INTO header VALUES(NULL,".$this->id().
            ",".( is_null( $shift_id       ) ? 'NULL' : $shift_id ).
            ",".( is_null( $run_id         ) ? 'NULL' : $run_id ).
            ",".( is_null( $relevance_time ) ? 'NULL' : LusiTime::to64from( $relevance_time )).")" );

        $this->connection->query (
            "INSERT INTO entry VALUES(NULL,(SELECT LAST_INSERT_ID()),NULL".
            ",".LusiTime::now()->to64().
            ",'".$author.
            "','".$content.
            "','".$content_type."')" );

        return $this->find_entry_by_ (
            'hdr_id = (SELECT h.id FROM header h, entry e'.
            ' WHERE h.id = e.hdr_id AND e.id = (SELECT LAST_INSERT_ID()))' );
    }
}
?>
