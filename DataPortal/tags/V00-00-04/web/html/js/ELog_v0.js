/*
 * ==============================================
 *  Application: e-Log, namespace of ids: elog-*
 * ==============================================
 */

function elog_create() {

	/* Add this anchor to access this object's variables from within
	 * for anonymous functions. Just using this.varname won't work
	 * due to a well known bug in JavaScript. */

	var that = this;

	/* ---------------------------
	 *  ELog specific environment
	 * ---------------------------
	 *
	 * NOTE: These variables should be initialized externally.
	 */
	this.author_account  = null;
	this.exper_id        = null;
	this.experiment_name = null;
	this.instrument_name = null;
	this.range_of_runs   = null;
	this.min_run         = null;
	this.max_run         = null;
	this.experiment_name = null;
	this.shifts          = new Array();
	this.runs            = new Array();

	/* ---------------------------------------------------------------------
	 *  Enable/disable input fields for posted message relevance parameters
	 * ---------------------------------------------------------------------
	 */
	function post_allow_relevance_inputs( shift, runnum, datepicker, time ) {

		var obj = null;

		obj = $('#elog-post-shift'      ); if( shift      ) obj.removeAttr('disabled'); else obj.attr('disabled', 'disabled');
		obj = $('#elog-post-runnum'     ); if( runnum     ) obj.removeAttr('disabled'); else obj.attr('disabled', 'disabled');
		obj = $('#elog-post-datepicker' ); if( datepicker ) obj.removeAttr('disabled'); else obj.attr('disabled', 'disabled');
		obj = $('#elog-post-time'       ); if( time       ) obj.removeAttr('disabled'); else obj.attr('disabled', 'disabled');
	}

	/* ----------------------------------------------
	 *  Initialize the form for posting new messages
	 * ----------------------------------------------
	 */
	this.post_init = function() {

		$('#elog-post-context-selector').buttonset();
		$('#elog-post-context-experiment').click(function(ev) {

			$('#elog-post-relevance-now'     ).attr      ('checked', 'checked');
			$('#elog-post-relevance-past'    ).removeAttr('checked');
			$('#elog-post-relevance-shift'   ).removeAttr('checked').attr('disabled', 'disabled');
			$('#elog-post-relevance-run'     ).removeAttr('checked').attr('disabled', 'disabled');
			$('#elog-post-relevance-selector').buttonset ('refresh');

			post_allow_relevance_inputs( false, false, false, false );
		});
		$('#elog-post-context-shift').click(function(ev) {

			$('#elog-post-relevance-now'     ).removeAttr('checked');
			$('#elog-post-relevance-past'    ).removeAttr('checked');
			$('#elog-post-relevance-shift'   ).attr      ('checked', 'checked').removeAttr('disabled');
			$('#elog-post-relevance-run'     ).removeAttr('checked'           ).attr      ('disabled', 'disabled');
			$('#elog-post-relevance-selector').buttonset ('refresh');

			post_allow_relevance_inputs( true, false, false, false );
		});
		$('#elog-post-context-run').click(function(ev) {

			$('#elog-post-relevance-now'     ).removeAttr('checked'           );
			$('#elog-post-relevance-past'    ).removeAttr('checked'           );
			$('#elog-post-relevance-shift'   ).removeAttr('checked'           ).attr      ('disabled', 'disabled');
			$('#elog-post-relevance-run'     ).attr      ('checked', 'checked').removeAttr('disabled');
			$('#elog-post-relevance-selector').buttonset ('refresh');

			post_allow_relevance_inputs( false, true, false, false );
		});

		$('#elog-post-relevance-selector').buttonset();
		$('#elog-post-relevance-now').click(function(ev) {
			$('#elog-post-datepicker').attr('disabled', 'disabled');
			$('#elog-post-time'      ).attr('disabled', 'disabled');
			var context = post_selected_context();
			if( context != 'shift' ) $('#elog-post-shift' ).attr('disabled', 'disabled');
			if( context != 'run'   ) $('#elog-post-runnum').attr('disabled', 'disabled');
		});
		$('#elog-post-relevance-past').click(function(ev) {
			$('#elog-post-datepicker').removeAttr('disabled');
			$('#elog-post-time'      ).removeAttr('disabled');
			var context = post_selected_context();
			if( context != 'shift' ) $('#elog-post-shift' ).attr('disabled', 'disabled');
			if( context != 'run'   ) $('#elog-post-runnum').attr('disabled', 'disabled');
		});
		$('#elog-post-relevance-shift').click(function(ev) {
			$('#elog-post-datepicker').attr      ('disabled', 'disabled');
			$('#elog-post-time'      ).attr      ('disabled', 'disabled');
			$('#elog-post-shift'     ).removeAttr('disabled');
			$('#elog-post-runnum'    ).attr      ('disabled', 'disabled');
		});
		$('#elog-post-relevance-run').click(function(ev) {
			$('#elog-post-datepicker').attr      ('disabled', 'disabled');
			$('#elog-post-time'      ).attr      ('disabled', 'disabled');
			$('#elog-post-runnum'    ).removeAttr('disabled');
		});

		$('#elog-post-datepicker').datepicker({
			showButtonPanel: true,
			dateFormat: 'yy-mm-dd'
		});
		$('#elog-post-datepicker').attr('disabled', 'disabled');
		$('#elog-post-time'      ).attr('disabled', 'disabled');
	
		$('#elog-tags-library-0').change(function(ev) {
			var selectedIndex = $('#elog-tags-library-0').attr('selectedIndex');
			if( selectedIndex == 0 ) return;
			$('#elog-tag-name-0'    ).val($('#elog-tags-library-0').val());
			$('#elog-tags-library-0').attr('selectedIndex', 0);
		});
		$('#elog-tags-library-1').change(function(ev) {
			var selectedIndex = $('#elog-tags-library-1').attr('selectedIndex');
			if( selectedIndex == 0 ) return;
			$('#elog-tag-name-1'    ).val($('#elog-tags-library-1').val());
			$('#elog-tags-library-1').attr('selectedIndex', 0);
		});
		$('#elog-tags-library-2').change(function(ev) {
			var selectedIndex = $('#elog-tags-library-2').attr('selectedIndex');
			if( selectedIndex == 0 ) return;
			$('#elog-tag-name-2'    ).val($('#elog-tags-library-2').val());
			$('#elog-tags-library-2').attr('selectedIndex', 0);
		});
	
		$('#elog-submit').button().click(function() {
	
			/* Validate the form and initialize hidden fields */

			$('#elog-form-post input[name="author_account"]').val(that.author_account);
			$('#elog-form-post input[name="id"]').val(that.exper_id);

			var urlbase = window.location.href;
			var idx = urlbase.indexOf( '&' );
			if( idx > 0 ) urlbase = urlbase.substring( 0, idx );
			$('#elog-form-post input[name="onsuccess"]').val(urlbase+'&page1=elog&page2=post');

			var context = post_selected_context();
			if( context == 'run' ) {

				if( that.min_run == null ) {
					$('#elog-post-runnum-error').text('the experiment has not taken any runs yet');
					return;
				}
				var run = $('#elog-post-runnum').val();
				if(( run < that.min_run ) || ( run > that.max_run )) {
					$('#elog-post-runnum-error').text('the run number is out of allowed range: '+that.min_run+'-'+that.max_run);
					return;
				}
				$('#elog-form-post input[name="run_id"]').val(that.runs[run]);

			} else if( context == 'shift' ) {

				var shift = $('#elog-post-shift').val();
				$('#elog-form-post input[name="shift_id"]').val(that.shifts[shift]);
			}
			if( post_selected_relevance() == 'past' ) {

				/* TODO: Check the syntax of the timestamp using regular expression
				 *       before submitting the request. The server side script will also
				 *       check its validity (applicability).
				 */
				var relevance_time = $('#elog-post-datepicker').val()+' '+$('#elog-post-time').val();
				$('#elog-form-post input[name="relevance_time"]').val(relevance_time);
			}

			$('#elog-form-post').trigger( 'submit' );
		});
	
		$('#elog-reset').button().click(function() {
			post_reset();
		});
	
		post_reset();
	};

	/* -------------------------------------
	 *  Reset the form to its initial state
	 * -------------------------------------
	 */
	function post_reset() {

		$('#elog-post-context-experiment').attr      ('checked', 'checked');
		$('#elog-post-context-shift'     ).removeAttr('checked');
		$('#elog-post-context-run'       ).removeAttr('checked');
		$('#elog-post-context-selector'  ).buttonset ('refresh');

		$('#elog-post-relevance-shift').attr('disabled', 'disabled');
		$('#elog-post-relevance-run'  ).attr('disabled', 'disabled');

		$('#elog-post-relevance-now'     ).attr      ('checked', 'checked');
		$('#elog-post-relevance-past'    ).removeAttr('checked');
		$('#elog-post-relevance-shift'   ).removeAttr('checked');
		$('#elog-post-relevance-run'     ).removeAttr('checked');
		$('#elog-post-relevance-selector').buttonset ('refresh');

		post_allow_relevance_inputs( false, false, false, false );
	}

	/* -------------------------------------
	 *  Add one more line for an attachment
	 * -------------------------------------
	 */
	var attachment_number = 0;
	this.post_add_attachment = function(e) {
		if( e.value != '' ) {
			attachment_number++;
			$( '#elog-post-attachments' ).append(
				'<input type="file" name="file2attach_'+attachment_number+'" onchange="elog.post_add_attachment(this)" />'+
				' '+
				'<input type="hidden" name="file2attach_'+attachment_number+'" value="" title="put an optional file description here" />'+
				'<br>'
			);
		}
	};

	/* --------------------------
	 *  Posting context selector
	 * --------------------------
	 */
	var post_id2context = new Array();
	post_id2context['elog-post-context-experiment']='experiment';
	post_id2context['elog-post-context-shift']='shift';
	post_id2context['elog-post-context-run']='run';

	function post_selected_context() {
		return post_id2context[$('#elog-post-context-selector input:checked').attr('id')];
	}

	/* ----------------------------
	 *  Posting relevance selector
	 * ----------------------------
	 */
	var post_id2relevance = new Array();
	post_id2relevance['elog-post-relevance-now']='now';
	post_id2relevance['elog-post-relevance-past']='past';
	post_id2relevance['elog-post-relevance-shift']='shift';
	post_id2relevance['elog-post-relevance-run']='run';

	function post_selected_relevance() {
		return post_id2relevance[$('#elog-post-relevance-selector input:checked').attr('id')];
	}

	/* ----------------------------------
	 *  Application initialization point
	 * ----------------------------------
	 */
	this.init = function() {

		$('#tabs-elog-subtabs').tabs();
		this.post_init();
	};
}

var elog = new elog_create();

/* End Of File */
