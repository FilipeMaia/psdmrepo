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

    private $label_color = ''; //  'color:#0071bc;'

    private $html;

    public function __construct( $x, $y,  $width=480, $height=null, $position='relative', $style=null, $highlight=false ) {
        $style_extra = is_null( $style ) ? '' : $style;
        $height_str = is_null( $height ) ? 'height:auto;' : "height:{$height}px;";
        $highlight_str = $highlight ? 'onmouseover="javascript:container_highlight(this)" onmouseout="javascript:container_unhighlight(this,document.bgColor)"' : '';
        $this->html =<<<HERE
<div style="position:{$position}; left:{$x}px; top:{$y}px; margin-left:0px; width:{$width}px; {$height_str} {$style_extra}" {$highlight_str} >
HERE;
    }

    public function label( $x, $y, $text, $bold=true, $color=null ) {
        $style_bold = $bold ? 'font-weight:bold;' : '';
        $style_color = is_null( $color ) ? '' : "color:{$color};";
        $this->html = $this->html.<<<HERE
<div style="position:absolute; left:{$x}px; top:{$y}px; text-align:left; {$this->label_color} {$style_bold} {$style_color}">
  {$text}
</div>
HERE;
        return $this;
    }
    public function label_1( $x, $y, $text, $width, $bold=true ) {
        $style_bold = $bold ? 'font-weight:bold;' : '';
        $this->html = $this->html.<<<HERE
<div style="background-color:#e0e0e0; width:{$width}px; padding:2px; position:absolute; left:{$x}px; top:{$y}px; text-align:left; {$this->label_color} {$style_bold}">
  {$text}
</div>
HERE;
        return $this;
    }

    public function value( $x, $y, $text, $color=null ) {
        $style_color = is_null( $color ) ? '' : "color:{$color};";
    	$this->html = $this->html.<<<HERE
<div style="position:absolute; left:{$x}px; top:{$y}px; text-align:left; {$style_color}">
  {$text}
</div>
HERE;
        return $this;
    }

    public function value_1( $x, $y, $text ) {
        $this->html = $this->html.<<<HERE
<div style="padding:2px; position:absolute; left:{$x}px; top:{$y}px; text-align:left;">
  {$text}
</div>
HERE;
        return $this;
    }

    public function value_input( $x, $y, $var, $text='', $title='', $size=null ) {
        $size_str = is_null( $size ) ? '' : "size=\"{$size}\"";
        //$width_str = is_null( $size ) ? '' : "width:{$size}px;";
        $this->html = $this->html.<<<HERE
<div style="position:absolute; left:{$x}px; top:{$y}px; text-align:left;">
  <input id="{$var}" type="text" name="{$var}" value="{$text}" style="padding:1px;" title="{$title} {$width_str}" {$size_str} />
</div>
HERE;
        return $this;
    }

    public function select_input( $x, $y, $var, $list, $default_selected='' ) {
        $this->html = $this->html.<<<HERE
<div style="position:absolute; left:{$x}px; top:{$y}px; text-align:left;">
  <select align="center" type="text" name="{$var}" style="padding:1px;">
HERE;
        foreach( $list as $l ) {
            if( $i == $default_selected )
                $this->html = $this->html."<option id=\"{$var}_default\" value=\"{$l}\">{$l}</option>";
            else
                $this->html = $this->html."<option value=\"{$l}\">{$l}</option>";
        }
        $this->html = $this->html.<<<HERE
  </select>
</div>
HERE;
        return $this;
    }

    public function checkbox_input( $x, $y, $var, $text, $checked=false ) {
        $checked_attr = $checked ? 'checked="checked"' : '';
        $this->html = $this->html.<<<HERE
<div style="position:absolute; left:{$x}px; top:{$y}px; text-align:left;">
  <input type="checkbox" name="{$var}" value="{$text}" {$checked_attr}/>
</div>
HERE;
        return $this;
    }

    public function radio_input( $x, $y, $group_name, $text, $checked=false ) {
        $checked_attr = $checked ? 'checked="checked"' : '';
        $this->html = $this->html.<<<HERE
<div style="position:absolute; left:{$x}px; top:{$y}px; text-align:left;">
  <input type="radio" name="{$group_name}" value="{$text}" {$checked_attr}>
</div>
HERE;
        return $this;
    }

    public function textarea( $x, $y, $text, $width=480, $height=128 ) {
        $this->html = $this->html.<<<HERE
<div style="position:absolute; left:{$x}px; top:{$y}px; text-align:left;">
  <textarea style="width:{$width}px; height:{$height}px; padding:4px;" disabled="disabled">{$text}</textarea>
</div>
HERE;
        return $this;
    }

    public function textarea_input( $x, $y, $var, $width=480, $height=128, $text='' ) {
        $this->html = $this->html.<<<HERE
<div style="position:absolute; left:{$x}px; top:{$y}px; text-align:left;">
  <textarea style="width:{$width}px; height:{$height}px; padding:4px;" name="{$var}">{$text}</textarea>
</div>
HERE;
        return $this;
    }

    public function textbox( $x, $y, $text, $width=480, $height=128 ) {
        $this->html = $this->html.<<<HERE
<div style="position:absolute; left:{$x}px; top:{$y}px; text-align:left;">
  <texbox style="width:{$width}px; height:{$height}px; padding:0px;">{$text}</texbox>
</div>
HERE;
        return $this;
    }

    public function container_1( $x, $y, $contents, $width=null, $height=null, $highlight=false, $bgcolor=null ) {
        $width_style  = is_null( $width  ) ? '' : "width:{$width}px;";
        $height_stlye = is_null( $height ) ? '' : "height:{$height}px;";
        $highlight_str = $highlight ? 'onmouseover="javascript:container_highlight(this)" onmouseout="javascript:container_unhighlight(this,document.bgColor)"' : '';
        $bgcolor_style = is_null( $bgcolor ) ? '' : 'background-color:'.$bgcolor.';';
        $this->html = $this->html.<<<HERE
<div style="position:absolute; left:{$x}px; top:{$y}px; text-align:left; overflow:auto; {$width_style} {$height_stlye} padding:0px; {$bgcolor_style}" {$highlight_str} >
  {$contents}
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

    public function button( $x, $y, $id, $name, $title=null ) {
        $title_attribute = is_null( $title ) ? '' : $title_attribute = 'title="'.$title.'"';
        $this->html = $this->html.<<<HERE
<div style="position:absolute; left:{$x}px; top:{$y}px; text-align:left;">
  <button id="{$id}" {$title_attribute}>{$name}</button>
</div>
HERE;
        return $this;
    }

    public function container( $x, $y, $id ) {
        $this->html = $this->html.<<<HERE
<div id="{$id}" style="position:absolute; left:{$x}px; top:{$y}px; text-align:left;">
</div>
HERE;
        return $this;
    }

    public function line( $x, $y, $width ) {
        $this->html = $this->html.<<<HERE
<div style="position:absolute; left:{$x}px; top:{$y}px; width:{$width}px; border-top-style:solid; border-top-width:1px;">
</div>
HERE;
        return $this;
    }
    public function html() {
        return $this->html.'</div>
';
    }
}
?>
