<?php
/*
 * The class representing experimental shifts.
 */
class LogBookShift {

    /* Data members
     */
    private $connection;
    private $experiment;

    public $attr;

    /* Constructor
     */
    public function __construct( $connection, $experiment, $attr ) {
        $this->connection = $connection;
        $this->experiment = $experiment;
        $this->attr = $attr;
    }

    /* Accessors
     */
    public function id() {
        return $this->attr['id']; }

    public function exper_id() {
        return $this->attr['exper_id']; }

    public function begin_time() {
        return LusiTime::from64( $this->attr['begin_time'] ); }

    public function end_time() {
        if( is_null( $this->attr['end_time'] )) return null;
        return LusiTime::from64( $this->attr['end_time'] ); }

    public function leader() {
        return $this->attr['leader']; }

    /* Close the open-ended shift
     */
    public function close( $end_time ) {

        if( !is_null( $this->attr['end_time'] ))
            throw new LogBookException(
                __METHOD__, "the shift is already closed" );

        /* Verify the value of the parameter
         */
        if( is_null( $end_time ))
            throw new LogBookException(
                __METHOD__, "end time can't be null" );

        if( 0 != $this->experiment->in_interval( $end_time ))
            throw new LogBookException(
                __METHOD__,
                "end time '".$end_time."' is out of experiment's interval" );

        /* Make the update
         */
        $end_time_64 = LusiTime::to64from( $end_time );
        $this->connection->query (
            'UPDATE "shift" SET end_time='.$end_time_64.
            ' WHERE exper_id='.$this->exper_id().' AND begin_time='.$this->attr['begin_time'] );

        /* Update the current state of the object
         */
        $this->attr['end_time'] = $end_time_64;
    }

    /**
     * Get a complete crew (names of all members including the leader)
     *
     * @return array
     */
    public function crew() {

        $list = array();

        $result = $this->connection->query(
            'SELECT member FROM shift_crew WHERE shift_id='.$this->id());

        $nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ ) {
            $member =  mysql_result( $result, $i );
            array_push( $list, $member );
            if( $member == $this->leader())
                $leader_in_crew = true;
        }
        if( !$leader_in_crew )
            array_push( $list, $this->leader());

        return $list;
    }
}
?>
