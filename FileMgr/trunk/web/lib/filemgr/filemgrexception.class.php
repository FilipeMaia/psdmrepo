<?php

namespace FileMgr;

require_once( 'filemgr.inc.php' );
/* 
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 * Class FileMgrException is a standard exception to be thrown by the API
 * to report wrongful conditions.
 *
 * The current implementation of the class doesn't have any extra functionality
 * on top of its base class. Therefore the sole role of the current class is
 * to provide an identification mechanism for recognizable non-standard
 * situations appearing within the API.
 *
 * @author gapon
 */
class FileMgrException extends \Exception {

    /*
     * Data members
     */
    protected $method;
    public    $errno = null;

    /**
     * Constructor
     *
     * @param string $message
     * @param int $code
     */
    public function __construct( $method, $message, $errno=null) {
        parent::__construct( $message );
        $this->method = $method;
        $this->errno = $errno;
    }

    /**
     * String representation of the exception
     *
     * @return string
     */
    public function __toString() {
        return __CLASS__ . "@".$this->method.": {$this->message}\n";
    }

    /**
     * HTML decorated string representation of the exception
     *
     * @return string
     */
    public function toHtml() {
        return "<b style='color:red'>".__CLASS__ . "</b> : <b>".$this->method."()</b> : <i>{$this->message}</i>\n";
    }
}
?>
