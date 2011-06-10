<script type="text/javascript">
$(function() {
	$('#es-menu').accordion({
		collapsible: true,
		autoHeight: false
	});
});
</script>

<style type="text/css">
  .es-m {
    padding: .2em;
    padding-right: 0;
    background: #5C9CCC url(/jquery/css/custom-theme/images/ui-bg_gloss-wave_55_5c9ccc_500x100.png) 50% 50% repeat-x;
    border: 0;
    -moz-border-radius: 5px;
    -webkit-border-radius: 5px;
    border-radius: 5px;
  }
  .es-m-item {
    border: 1px solid #a0a0a0;
    padding:4px;
    width:100%;
    background-color:#e0e0e0;
    cursor:pointer;
    font-weight:bold;
  }
  div.es-m-item:hover {
    background-color: #ffffff;
  }
  .m-item-first {
    float:left;
  }
  .m-item {
    float:left;
  }
  .m-item-space {
    float:left;
    /*width:100%;*/
  }
  .m-item-last {
    float:right;
  }
  .m-item-end {
    clear:both;
  }
  
</style>

<div style="margin-top:20px;">

  <div style="float:left; width:10em;">

    <div id="es-menu" class="es-m">

      <h3><a href="#">Recent (Live)</a></h3>
      <div>
          <div class="es-m-item">20</div>
          <div class="es-m-item">100</div>
          <div class="es-m-item">shift</div>
          <div class="es-m-item">24 hrs</div>
          <div class="es-m-item">7 days</div>
          <div class="es-m-item">everything</div>
      </div>

	  <h3><a href="#">Post</a></h3>
      <div>
        <ul>
          <li>for experiment</li>
          <li>for shift</li>
          <li>for run</li>
        </ul>
      </div>

	  <h3><a href="#">Search</a></h3>
      <div></div>

	  <h3><a href="#">Browse</a></h3>
      <div></div>

	  <h3><a href="#">Shifts</a></h3>
      <div></div>

	  <h3><a href="#">Runs</a></h3>
      <div></div>

	  <h3><a href="#">Subscribe</a></h3>
      <div></div>

    </div>

  </div>

  <div style="float:right;">

  <div style="width:100%; border:1px solid;">
    <div class="m-item-first">First</div>
    <div class="m-item">Second</div>
    <div class="m-item">Third</div>
    <div class="m-item-space"></div>
    <div class="m-item-last">Last</div>
    <div class="m-item-end"></div>
  </div>


  </div>

  <div style="clear:both;"></div>

</div>