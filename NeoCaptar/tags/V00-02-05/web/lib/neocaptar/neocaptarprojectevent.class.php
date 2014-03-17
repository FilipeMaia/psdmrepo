<?php

namespace NeoCaptar;

require_once( 'neocaptar.inc.php' );

/**
 * Class NeoCaptarProjectEvent represents history events for projects.
 *
 * @author gapon
 */
class NeoCaptarProjectEvent extends NeoCaptarEvent {

   /* Data members
     */
    private $project;

    /* Constructor
     */
    public function __construct ($connection, $project, $attr) {
        parent::__construct($connection,'project',$project->id(),$attr);
        $this->project = $project;
    }

    /*
     * ======================
     *   OBJECT ATTRIBUTES
     * ======================
     */
    public function project () { return $this->project; }
}
?>
