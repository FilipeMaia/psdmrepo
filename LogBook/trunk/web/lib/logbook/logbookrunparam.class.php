<?php

namespace LogBook;

require_once( 'logbook.inc.php' );

class LogBookRunParam {

    /* Data members
     */
    private $connection;
    private $experiment;

    public $attr;

    /** Constructor
     */
    public function __construct ( $connection, $experiment, $attr ) {
        $this->connection = $connection;
        $this->experiment = $experiment;
        $this->attr = $attr;
    }

    /* Accessors
     */
    public function parent () {
        return $this->experiment; }

    public function id () {
        return $this->attr['id']; }

    public function name () {
        return $this->attr['param']; }

    public function exper_id () {
        return $this->attr['exper_id']; }

    public function type_name () {
        return $this->attr['type']; }

    public function description () {
        return $this->attr['descr']; }
}
?>
