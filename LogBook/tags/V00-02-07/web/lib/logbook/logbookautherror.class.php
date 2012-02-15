<?php

namespace LogBook;

require_once( 'logbook.inc.php' );

/* 
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 * Class LogBookAuthError is an exception to be thrown by the API
 * to report wrongful problems with user authentication/authorization.
 *
 * @author gapon
 */
class LogBookAuthError extends Exception {

    /*
     * Data members
     */
    protected $context;

    /**
     * Constructor
     *
     * @param string $message
     * @param string $context
     */
    public function __construct( $context, $message) {
        parent::__construct( $message );
        $this->context = $context;
    }

    /**
     * String representation of the exception
     *
     * @return string
     */
    public function __toString() {
        return __CLASS__ . "@".$this->context.": {$this->message}\n";
    }

    /**
     * HTML decorated string representation of the exception
     *
     * @return string
     */
    public function toHtml() {
        return "<b style='color:red'>".__CLASS__ . "</b> : <b>".$this->context."()</b> : <i>{$this->message}</i>\n";
    }
}
?>
