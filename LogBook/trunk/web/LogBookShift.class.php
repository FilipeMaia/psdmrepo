<?php
/*
 * The class representing experimental shifts.
 * 
 * TODO:
 * - migrate from die() to exceptions
 * - add JSON serializer
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
    public function exper_id() {
        return $this->attr['exper_id']; }

    public function begin_time() {
        return LogBookTime::from64( $this->attr['begin_time'] ); }

    public function end_time() {
        if( is_null( $this->attr['end_time'] )) return null;
        return LogBookTime::from64( $this->attr['end_time'] ); }

    public function leader() {
        return $this->attr['leader']; }

    /* Close the open-ended shift
     */
    public function close( $end_time ) {

        if( !is_null($this->attr['end_time']))
            die( "the shift is already closed" );

        /* Verify the value of the parameter
         */
        if( is_null( $end_time ))
            die( "end time can't be null");
        if( 0 != $this->experiment->in_interval( $end_time ))
            die( "incorrect end time - it falls off the experiment's intervall" );

        /* Make the update
         */
        $end_time_64 = LogBookTime::to64from( $end_time );
        $this->connection->query(
            'UPDATE "shift" SET end_time='.$end_time_64.
            ' WHERE exper_id='.$this->exper_id().' AND begin_time='.$this->attr['begin_time'] )
            or die ("failed to close the shift: ".mysql_error());

        /* Update the current state of the object
         */
        $this->attr['end_time'] = $end_time_64;
        return true;
    }
}
?>
