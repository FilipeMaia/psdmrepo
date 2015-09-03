<?php

namespace RegDB;

require_once( 'regdb.inc.php' );

/**
 * Class RegDBExperimentParam an abstraction for experiment parameters.
 *
 * TODO: Consider reimplementing methods value() and description()
 *       to load the requested information on demand if the dictionary
 *       of attributes won't have the corresponding keys.
 *
 * @author gapon
 */
class RegDBExperimentParam {

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

    public function exper_id () {
        return $this->attr['exper_id']; }

    public function name () {
        return $this->attr['param']; }

    public function value () {
        return $this->attr['val']; }

    public function description () {
        return $this->attr['descr']; }
}
?>
