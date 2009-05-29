<?php
class LogBookConnection {

    /* Data members
     */
    private $link;
    private $host;
    private $user;
    private $password;
    private $database;

    /* Constructor
     *
     * NOTE: The constructor won't make any actual connection
     *       attempts. This will be deffered to operations
     *       dealing with queries, transactions, etc.
     *
     * TODO: Add explicit transction management.
     */
    public function __construct ( $host, $user, $password, $database ) {
        $this->host     = $host;
        $this->user     = $user;
        $this->password = $password;
        $this->database = $database;
    }

    public function __destruct () {
        if( isset( $this->link ))
            mysql_close( $this->link );
    }

    public function query ( $sql ) {
        $this->connect();
        $result = mysql_query( $sql );
        if( !$result )
            throw new LogBookException(
                __METHOD__,
                "MySQL error: ".mysql_error().', in query: '.$sql );
        return $result;
    }

    private function connect () {
        if( !isset( $this->link )) {
            $this->link = mysql_connect( $this->host, $this->user, $this->password );
            if( !$this->link )
                throw new LogBookException(
                    __METHOD__,
                    "MySQL error: ".mysql_error().", in function: mysql_connect" );

            if( !mysql_select_db( $this->database ))
                throw new LogBookException(
                    __METHOD__,
                    "MySQL error: ".mysql_error().", in function: mysql_select_db" );

            $sql = "SET SESSION SQL_MODE='ANSI'";
            if( !mysql_query( $sql ))
                throw new LogBookException(
                    __METHOD__,
                    "MySQL error: ".mysql_error().', in query: '.$sql );
        }
    }
}
?>
