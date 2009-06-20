<?php
/* 
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 * Class RegDBHtml provides utilities for HTML formatting
 *
 * @author gapon
 */
class RegDBHtml {

    private $html;

    public function __construct( $x, $y,  $width=480, $height=320, $position='relative' ) {
        $this->html = "<div style=\"position:{$position}; left:{$x}px; top:{$y}px; margin-left:0px; width:{$width}px; height:{$height}px;\">";
    }

    public function label( $x, $y, $text ) {
        $this->html = $this->html."
<div style=\"position:absolute; left:{$x}px; top:{$y}px; text-align:left; color:#0071bc; font-weight:bold;\">
  {$text}
</div>";
        return $this;
    }

    public function value( $x, $y, $text ) {
        $this->html = $this->html."
<div style=\"position:absolute; left:{$x}px; top:{$y}px; text-align:left;\">
  {$text}
</div>";
        return $this;
    }

    public function textarea( $x, $y, $text, $width=480, $height=128 ) {
        $this->html = $this->html."
<div style=\"position:absolute; left:{$x}px; top:{$y}px; text-align:left;\">
  <textarea style=\"width:{$width}px; height:{$height}px; padding:4px;\"  disabled=\"disabled\">{$text}</textarea></td>
</div>";
        return $this;
    }
    public function html() {
        return $this->html.'</div>';
    }
}
?>
