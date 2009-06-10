<?php
class LogBookRunVal {

    /* Data members
     */
    private $connection;
    private $run;

    public $attr;

    /* Constructor
     */
    public function __construct( $connection, $run, $attr ) {
        $this->connection = $connection;
        $this->run = $run;
        $this->attr = $attr;
    }

    /* Accessors
     */
    public function parent() {
        return $this->run; }

    public function run_id() {
        return $this->attr['run_id']; }

    public function param_id() {
        return $this->attr['param_id']; }

    public function source() {
        return $this->attr['source']; }

    public function updated() {
        return LusiTime::from64( $this->attr['updated'] ); }

    public function value() {
        return $this->attr['val']; }
}
?>
