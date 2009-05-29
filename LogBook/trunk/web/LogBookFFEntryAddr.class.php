<?php
/* 
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 * Class LogBookFFEntryAddr represents full addresses of
 * free-form entries in the database. Each address has:
 *
 * - an identifier of the entry header
 * - an optional identifier of a apscific entry within a hierarchy
 *
 * If the optional identifier is set to null then the root entry
 * directly attached to the header is assumed.
 *
 * @author gapon
 */
class LogBookFFEntryAddr {

    /* Data members
     */
    public $attr;

    /* Constructor from an associative array
     */
    public function __construct ( $attr ) {
        $this->attr = $attr;
    }

    /* Constructor from a pair of identifiers
     */
    public function __construct ( $hdr_id, $id=null ) {
        $this->attr = array(
            'hdr_id' => $hdr_id,
            'id'     => $id );
    }

    /* Accessors
     */
    public function hdr_id() {
        return $this->attr['hdr_id']; }

    public function id() {
        return $this->attr['id']; }
}
?>
