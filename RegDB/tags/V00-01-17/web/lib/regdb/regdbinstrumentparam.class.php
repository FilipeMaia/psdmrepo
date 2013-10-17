<?php

namespace RegDB;

require_once( 'regdb.inc.php' );

/**
 * Class RegDBInstrumentParam an abstraction for instrument parameters.
 *
 * TODO: Consider reimplementing methods value() and description()
 *       to load the requested information on demand if the dictionary
 *       of attributes won't have the corresponding keys.
 *
 * @author gapon
 */
class RegDBInstrumentParam {

    /* Data members
     */
    private $connection;
    private $instrument;

    public $attr;

    /* Constructor
     */
    public function __construct ( $connection, $instrument, $attr ) {
        $this->connection = $connection;
        $this->instrument = $instrument;
        $this->attr = $attr;
    }

    public function parent () {
        return $this->instrument; }

    public function instr_id () {
        return $this->attr['instr_id']; }

    public function name () {
        return $this->attr['param']; }

    public function value () {
        return $this->attr['val']; }

    public function description () {
        return $this->attr['descr']; }
}
?>
