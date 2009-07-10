<?php
/**
 * Class RegDBRun an abstraction for experimental runs.
 *
 * @author gapon
 */
class RegDBRun {

    /* Data members
     */
    private $connection;
    private $experiment;

    public $attr;

    /* Constructor
     */
    public function __construct ( $connection, $experiment, $attr ) {
        $this->connection = $connection;
        $this->experiment = $experiment;
        $this->attr = $attr;
    }

    public function parent () {
        return $this->experiment; }

    public function num () {
        return $this->attr['num']; }

    public function request_time () {
        return LusiTime::from64( $this->attr['request_time'] ); }
}
?>
