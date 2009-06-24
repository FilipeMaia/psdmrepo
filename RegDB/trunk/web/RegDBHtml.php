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
        $this->html =<<<HERE
<div style="position:{$position}; left:{$x}px; top:{$y}px; margin-left:0px; width:{$width}px; height:{$height}px;">
HERE;
    }

    public function label( $x, $y, $text ) {
        $this->html = $this->html.<<<HERE
<div style="position:absolute; left:{$x}px; top:{$y}px; text-align:left; color:#0071bc; font-weight:bold;">
  {$text}
</div>
HERE;
        return $this;
    }

    public function value( $x, $y, $text ) {
        $this->html = $this->html.<<<HERE
<div style="position:absolute; left:{$x}px; top:{$y}px; text-align:left;">
  {$text}
</div>
HERE;
        return $this;
    }

    public function value_input( $x, $y, $var, $text='' ) {
        $this->html = $this->html.<<<HERE
<div style="position:absolute; left:{$x}px; top:{$y}px; text-align:left;">
  <input id="{$var}" type="text" name="{$var}" value="{$text}" style="padding:1px;" />
</div>
HERE;
        return $this;
    }

    public function select_input( $x, $y, $var, $list ) {
        $this->html = $this->html.<<<HERE
<div style="position:absolute; left:{$x}px; top:{$y}px; text-align:left;">
  <select align="center" type="text" name="{$var}" style="padding:1px;">
HERE;
        foreach( $list as $l )
            $this->html = $this->html."<option>{$l}</option>";
        $this->html = $this->html.<<<HERE
  </select>
</div>
HERE;
        return $this;
    }

    public function textarea( $x, $y, $text, $width=480, $height=128 ) {
        $this->html = $this->html.<<<HERE
<div style="position:absolute; left:{$x}px; top:{$y}px; text-align:left;">
  <textarea style="width:{$width}px; height:{$height}px; padding:4px;" disabled="disabled">{$text}</textarea></td>
</div>
HERE;
        return $this;
    }

    public function textarea_input( $x, $y, $var, $width=480, $height=128, $text='' ) {
        $this->html = $this->html.<<<HERE
<div style="position:absolute; left:{$x}px; top:{$y}px; text-align:left;">
  <textarea style="width:{$width}px; height:{$height}px; padding:4px;" name="{$var}">{$text}</textarea></td>
</div>
HERE;
        return $this;
    }

    public function hidden_action( $name, $value='' ) {
        $this->html = $this->html.<<<HERE
<input type="hidden" name="{$name}" value="{$value}" />
HERE;
        return $this;
    }
    public function html() {
        return $this->html.'</div>
';
    }
}
?>
