<?php
require_once('LogBook.inc.php');

/* Class for constructing HTML tables of specified configurations,
 * which includes the following parameters:
 * - an array of collumn names
 * - an array of the corresponding (to collumns) keys to be used when
 *   extracting data for table rows.
 */
class LogBookTestTable {

    /* Data members
     */
    private $cols;
    private $keys;
    private $css_class;

    /* Constructor
     */
    public function __construct( $cols, $keys, $css_class ) {
        if(count($cols) != count($keys))
            die("illegal parameters to the contsructor");
        $this->cols = $cols;
        $this->keys = $keys;
        $this->css_class = $css_class;
    }

    public function begin() {
        echo <<<HERE
<table cellpadding="3"  border="0" class="$this->css_class">
    <thead style="color:#0071bc;">
HERE;
        foreach($this->cols as $c)
            echo <<<HERE
        <th align="left">
            &nbsp;<b>$c</b>&nbsp;</th>
HERE;
        echo <<<HERE
    </thead>
    <tbody>
        <tr>
HERE;
        foreach($this->cols as $c)
            echo <<<HERE
            <td><hr></td>
HERE;
        echo <<<HERE
        </tr>
HERE;
    }

    public function row($attr) {
        echo <<<HERE
        <tr>
HERE;
        foreach($this->keys as $k) {
            $v = $attr[$k];
            echo <<<HERE
            <td>&nbsp;$v&nbsp;</td>
HERE;
        }
    echo <<<HERE
        </tr>
HERE;
    }

    public function end() {
        echo <<<HERE
    </tbody>
</table>
HERE;
    }

    /*
     * Display a complete table instance from the input array
     */
    public function show( $list, $title=null ) {
        if( !is_null($title))
            echo <<<HERE
$title
HERE;
        $this->begin();
        foreach( $list as $e )
            $this->row($e->attr);
        $this->end();
    }

    /* Factory methods for predefined tables */

    public static function Experiment($css_class='table_2') {
        return new LogBookTestTable(
            array("Id", "Name", "Begin Time", "End Time"),
            array("id", "name", "begin_time", "end_time"),
            $css_class );
    }
    public static function Shift($css_class='table_4') {
        return new LogBookTestTable(
            array("Experiment Id", "Begin Time", "End Time", "Shift Leader"),
            array("exper_id",      "begin_time", "end_time", "leader"),
            $css_class );
    }
    public static function RunParam($css_class='table_4') {
        return new LogBookTestTable(
            array("Id", "Name",  "Experiment Id", "Type", "Description"),
            array("id", "param", "exper_id",      "type", "descr"),
            $css_class );
    }
    public static function Run($css_class='table_4') {
        return new LogBookTestTable(
            array("Id", "Number",  "Experiment Id", "Begin Time", "End Time"),
            array("id", "num",     "exper_id",      "begin_time", "end_time"),
            $css_class );
    }
    public static function RunVal($css_class='table_6') {
        return new LogBookTestTable(
            array("Run Id", "Param Id",  "Source", "Updated", "Value"),
            array("run_id", "param_id",  "source", "updated", "val"),
            $css_class );
    }
}
?>
