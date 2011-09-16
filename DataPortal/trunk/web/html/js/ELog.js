/*
 * ===============================================
 *  Application: e-Log
 *  DOM namespace of classes and identifiers: el-
 *  JavaScript global names begin with: elog
 * ===============================================
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
	this.author  = null;
	this.exp_id  = null;
	this.exp     = null;
	this.instr   = null;
	this.shifts  = new Array();
	this.editor  = false;
	this.post_onsuccess = null;
	this.used_tags = new Array();

	/* The context for v-menu items
	 */
	var context2_default = {
		'recent'      : '',
		'post'        : 'experiment',
		'search'      : '',
		'shifts'      : '',
		'runs'        : '',
		'attachments' : '',
		'subscribe'   : ''
	};
	this.name      = 'elog';
	this.full_name = 'e-Log';
	this.context1  = 'recent';
	this.context2  = context2_default[this.context1];
	this.select_default = function() { this.select(this.context1, this.context2); };
	this.select = function(ctx1, ctx2) {

		var just_initialized = this.init();

		var prev_context1 = this.context1;
		this.context1 = ctx1;

		var prev_context2 = this.context2;
		this.context2 = ctx2 == null ? context2_default[ctx1] : ctx2;


		this.context3 ='';
		if(this.context1 == 'recent') {

			// Do nothing if nothing has changed since the previous call to the function,
			// such as: fresh initialization, change in the slected context, subcontext, etc.
			//
			if( just_initialized || (prev_context1 != this.context1 )) {
				this.live_dim_all_highlights();
				this.live_message_viewer.reload(live_selected_runs(), this.live_selected_range());
			}
		} else if(this.context1 == 'post') {

			$('#elog-form-post input[name="scope"]').val(this.context2);
			if(this.context2 == 'experiment') {

				$('#el-p-message4experiment').removeClass('hidden' ).addClass('visible');
				$('#el-p-message4shift'     ).removeClass('visible').addClass('hidden');
				$('#el-p-message4run'       ).removeClass('visible').addClass('hidden');

			} else if(this.context2 == 'shift') {

				$('#el-p-message4experiment').removeClass('visible').addClass('hidden');
				$('#el-p-message4shift'     ).removeClass('hidden' ).addClass('visible');
				$('#el-p-message4run'       ).removeClass('visible').addClass('hidden');

			} else if(this.context2 == 'run') {

				$('#el-p-message4experiment').removeClass('visible').addClass('hidden');
				$('#el-p-message4shift'     ).removeClass('visible').addClass('hidden');
				$('#el-p-message4run'       ).removeClass('hidden' ).addClass('visible');
			}
			this.post_reset();
		}
	};
    this.report_error = function(msg) {
        $('#popupdialogs').html( '<p><span class="ui-icon ui-icon-alert" style="float:left;"></span>'+msg+'</p>' );
        $('#popupdialogs').dialog({
        	resizable: false,
        	modal: true,
        	buttons: {
        		'Ok': function() { $(this).dialog('close'); }
        	},
        	title: 'Error'
        });
    };

	/* ----------------------------------------------
	 *  Initialize the form for posting new messages
	 * ----------------------------------------------
	 */
	this.post_init = function() {

		$('#el-p-context-experiment').click(function(ev) {

			$('#el-p-relevance-now'     ).attr      ('checked', 'checked');
			$('#el-p-relevance-past'    ).removeAttr('checked');
			$('#el-p-relevance-shift'   ).removeAttr('checked').attr('disabled', 'disabled');
			$('#el-p-relevance-run'     ).removeAttr('checked').attr('disabled', 'disabled');
			$('#el-p-relevance-selector').buttonset ('refresh');

		});
		$('#el-p-context-shift').click(function(ev) {

			$('#el-p-relevance-now'     ).removeAttr('checked');
			$('#el-p-relevance-past'    ).removeAttr('checked');
			$('#el-p-relevance-shift'   ).attr      ('checked', 'checked').removeAttr('disabled');
			$('#el-p-relevance-run'     ).removeAttr('checked'           ).attr      ('disabled', 'disabled');
			$('#el-p-relevance-selector').buttonset ('refresh');

		});
		$('#el-p-context-run').click(function(ev) {

			$('#el-p-relevance-now'     ).removeAttr('checked'           );
			$('#el-p-relevance-past'    ).removeAttr('checked'           );
			$('#el-p-relevance-shift'   ).removeAttr('checked'           ).attr      ('disabled', 'disabled');
			$('#el-p-relevance-run'     ).attr      ('checked', 'checked').removeAttr('disabled');
			$('#el-p-relevance-selector').buttonset ('refresh');

		});

		$('#el-p-relevance-selector').buttonset();
		$('#el-p-relevance-now').click(function(ev) {
			$('#el-p-datepicker').attr('disabled', 'disabled');
			$('#el-p-time'      ).attr('disabled', 'disabled');
		});
		$('#el-p-relevance-past').click(function(ev) {
			$('#el-p-datepicker').removeAttr('disabled');
			$('#el-p-time'      ).removeAttr('disabled');
		});
		$('#el-p-relevance-shift').click(function(ev) {
			$('#el-p-datepicker').attr('disabled', 'disabled');
			$('#el-p-time'      ).attr('disabled', 'disabled');
		});
		$('#el-p-relevance-run').click(function(ev) {
			$('#el-p-datepicker').attr('disabled', 'disabled');
			$('#el-p-time'      ).attr('disabled', 'disabled');
		});

		$('#el-p-datepicker').datepicker({
			showButtonPanel: true,
			dateFormat: 'yy-mm-dd'
		});
		$('#el-p-datepicker').attr('disabled', 'disabled');
		$('#el-p-time'      ).attr('disabled', 'disabled');
	
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
	
		$('#elog-post-submit').button().click(function() {
	
			var urlbase = window.location.href;
			var idx = urlbase.indexOf( '&' );
			if( idx > 0 ) urlbase = urlbase.substring( 0, idx );

			// Make sure there is anything to submit. Do not allow submitting
			// an empty message.
			//
			if( $('#elog-form-post textarea[name="message_text"]').val() == '' ) {
				that.report_error('Can not post the empty message. Please put some text into the message box.');
				return;
			}

			if( that.context2 == 'run' ) {
				$('#elog-form-post input[name="run_num"]').val($('#el-p-runnum').val());
			} else if( that.context2 == 'shift' ) {
				$('#elog-form-post input[name="shift_id"]').val(that.shifts[$('#el-p-shift').val()]);
			}
			switch( post_selected_relevance()) {

			case 'now':
				$('#elog-form-post input[name="relevance_time"]').val('now');
				break;

			case 'past':
				/* TODO: Check the syntax of the timestamp using regular expression before submitting
				 * the request. The server side script will also check its validity/applicability.
				 */
				$('#elog-form-post input[name="relevance_time"]').val($('#el-p-datepicker').val()+' '+$('#el-p-time').val());
				break;
			}

			/* Submit the new message using the JQuery AJAX POST plug-in,
			 * which also allow uploading files w/o reloading the current page.
			 *
			 * NOTE: We aren't refreshing the list of messages because we're relying on
			 *       the live display.
			 */
			$('#elog-form-post').ajaxSubmit({
				success: function(data) {
					if( data.Status != 'success' ) { that.report_error(data.Message); return; }
					that.post_reset();

					// If the parent provided a call back then tell the parent
					// that we have a new message.
					//
					if( that.post_onsuccess != null ) that.post_onsuccess();
				},
				error: function() {	that.report_error('The request can not go through due a failure to contact the server.'); },
				dataType: 'json'
			});
		});

		$('#elog-post-reset').button().click(function() { that.post_reset(); });

		this.post_reset();
	};
	this.simple_post4experiment = function(text2post) {
		this.post_reset();
		$('#elog-form-post textarea[name="message_text"]').val(text2post);
	};

	/* ------------------------------------------
	 *  Reset the post form to its initial state
	 * ------------------------------------------
	 */
	this.post_reset = function() {

		$('#elog-form-post textarea[name="message_text"]').val('');
		$('#elog-form-post input[name="author_account"]').val(this.author);
		$('.elog-tag-name').val('');
		if(that.context2 == 'experiment') {

			$('#el-p-relevance-now'     ).attr      ('checked', 'checked');
			$('#el-p-relevance-past'    ).removeAttr('checked');
			$('#el-p-relevance-shift'   ).removeAttr('checked').attr('disabled', 'disabled');
			$('#el-p-relevance-run'     ).removeAttr('checked').attr('disabled', 'disabled');
			$('#el-p-relevance-selector').buttonset ('refresh');

		} else if(that.context2 == 'shift') {

			$('#el-p-relevance-now'     ).removeAttr('checked');
			$('#el-p-relevance-past'    ).removeAttr('checked');
			$('#el-p-relevance-shift'   ).attr      ('checked', 'checked').removeAttr('disabled');
			$('#el-p-relevance-run'     ).removeAttr('checked'           ).attr      ('disabled', 'disabled');
			$('#el-p-relevance-selector').buttonset ('refresh');

		} else if(that.context2 == 'run') {

			$('#el-p-relevance-now'     ).removeAttr('checked'           );
			$('#el-p-relevance-past'    ).removeAttr('checked'           );
			$('#el-p-relevance-shift'   ).removeAttr('checked'           ).attr      ('disabled', 'disabled');
			$('#el-p-relevance-run'     ).attr      ('checked', 'checked').removeAttr('disabled');
			$('#el-p-relevance-selector').buttonset ('refresh');
		}
		$('#el-p-datepicker').attr('disabled', 'disabled');
		$('#el-p-time'      ).attr('disabled', 'disabled');
		$('#el-p-as').html(
'<div>'+
'  <input type="file" name="file2attach_0" onchange="elog.post_add_attachment()" />'+
'  <input type="hidden" name="file2attach_0" value="" />'+
'</div>'
		);
	};

	/* -------------------------------------
	 *  Add one more line for an attachment
	 * -------------------------------------
	 */
	this.post_add_attachment = function() {
		var num = $('#el-p-as > div').size();
		$('#el-p-as').append(
'<div>'+
' <input type="file" name="file2attach_'+num+'" onchange="elog.post_add_attachment()" />'+
' <input type="hidden" name="file2attach_'+num+'" value="" />'+
'</div>'
		);
	};

	/* ----------------------------
	 *  Posting relevance selector
	 * ----------------------------
	 */
	var post_id2relevance = new Array();
	post_id2relevance['el-p-relevance-now']='now';
	post_id2relevance['el-p-relevance-past']='past';
	post_id2relevance['el-p-relevance-shift']='shift';
	post_id2relevance['el-p-relevance-run']='run';

	function post_selected_relevance() {
		return post_id2relevance[$('#el-p-relevance-selector input:checked').attr('id')];
	}

	/* -------------------------------------------
	 *  Initialize the 'Live' message display tab
	 * -------------------------------------------
	 */
	this.live_message_viewer = null;
	this.live_init = function() {

		this.live_message_viewer = new elog_message_viewer_create('elog.live_message_viewer', this, 'el-l');

		$('#el-l-rs-selector').buttonset().change(function() {
			that.live_message_viewer.dim_day();
			that.live_dim_all_highlights();
			that.live_message_viewer.reload(live_selected_runs(), that.live_selected_range());
		});
		$('#el-l-mctrl').find('select[name="messages"]').change(function() {
			that.live_message_viewer.dim_day();
			that.live_dim_all_highlights();
			that.live_message_viewer.reload(live_selected_runs(), that.live_selected_range());
		});
		$('#el-l-refresh-selector').buttonset().change(function() {
			if(live_selected_refresh()) {
				$('#el-l-refresh-interval').removeAttr('disabled');
				that.live_schedule_refresh();
			} else {
				$('#el-l-refresh-interval').attr('disabled', true);
				that.live_stop_refresh();
			}
		});
		$('#el-l-refresh').button().click(function() {
			/*
			that.live_dim_all_highlights();
			that.live_message_viewer.refresh(live_selected_runs(), that.live_highlight);
			*/
			//that.live_stop_refresh();
			that.live_message_viewer.reload(live_selected_runs(), that.live_selected_range());
			//that.live_schedule_refresh();
			//that.live_start_highlight_timer();
		});
		$('#el-l-refresh-interval').change(function(ev) {
			that.live_stop_refresh();
			that.live_schedule_refresh();
		});

		this.live_message_viewer.reload(live_selected_runs(), this.live_selected_range());
		this.live_schedule_refresh();
		this.live_start_highlight_timer();
	};


	/* -------------------------------------------------
	 * Live display timers and relevant periodic actions
	 * -------------------------------------------------
	 */
	function live_refresh_interval() {
		return 1000*parseInt($('#el-l-refresh-interval').val());
	}

	this.live_highlight_interval = 30*1000;
	this.live_highlighted = new Array();
	this.live_highlight_timer = null;
	this.live_highlight = function(id) {
		that.live_highlighted[id] = that.live_highlight_interval;
		$(id).addClass('el-l-m-highlight');
	}
	this.live_start_highlight_timer = function() {
	    this.live_highlight_timer = window.setTimeout('elog.live_highlight_actions()',live_refresh_interval());
	};
	this.live_highlight_actions = function() {
		for(var id in this.live_highlighted) {
			this.live_highlighted[id] -= live_refresh_interval();
			if(this.live_highlighted[id] <= 0) {
				$(id).removeClass('el-l-m-highlight');
				delete this.live_highlighted[id];
			}
		}
		this.live_start_highlight_timer();
	};
	this.live_dim_all_highlights = function() {
		for(var id in this.live_highlighted) {
			this.live_highlighted[id] -= live_refresh_interval();
			$(id).removeClass('el-l-m-highlight');
			delete this.live_highlighted[id];
		}
	};
	this.live_stop_highlight_timer = function() {
		this.live_dim_all_highlights();
	    if(this.live_highlight_timer != null) {
	        window.clearTimeout(this.live_highlight_timer);
	        this.live_highlight_timer = null;
	    }
	};

	this.live_refresh_timer = null;
	this.live_schedule_refresh = function() {
	    this.live_refresh_timer = window.setTimeout('elog.live_refresh_actions()',live_refresh_interval());
	};
	this.live_stop_refresh = function() {
	    if(this.live_refresh_timer != null) {
	        window.clearTimeout(this.live_refresh_timer);
	        this.live_refresh_timer = null;
	    }
	};
	this.live_refresh_actions = function() {
		this.live_message_viewer.refresh(live_selected_runs(), this.live_highlight);
		this.live_schedule_refresh();
	};

	/* -------------------------
	 *  Live display: selectors
	 * -------------------------
	 */
	var live_id2runs = new Array();
	live_id2runs['el-l-rs-on']=1;
	live_id2runs['el-l-rs-off']=0;
	function live_selected_runs() {
		return live_id2runs[$('#el-l-rs-selector input:checked').attr('id')];
	}

	this.live_selected_range = function() {
		var str = $('#el-l-mctrl').find('select[name="messages"]').val();
		switch(str) {
		case '20': return str;
		case '100': return str;
		case 'shift': return '12h';
		case 'day': return '24h';
		case 'week': return '7d';
		}
		return '';
	};

	var live_id2refresh = new Array();
	live_id2refresh['el-l-refresh-on']=1;
	live_id2refresh['el-l-refresh-off']=0;
	function live_selected_refresh() {
		return live_id2refresh[$('#el-l-refresh-selector input:checked').attr('id')];
	}

	/* --------------------------------------------
	 *  Initialize the form for searching messages
	 * --------------------------------------------
	 */
	this.search_message_viewer = null;
	this.search_init = function() {
		this.search_message_viewer = new elog_message_viewer_create('elog.search_message_viewer', this, 'el-s');
		$('#elog-search-submit').button().click(function() { that.search(); });
		$('#elog-search-reset').button().click(function() {	that.search_reset(); });
		$('#elog-form-search').find('input[name="text2search"]').keyup(function(e) { if( e.keyCode == 13 ) that.search(); });
		$('#elog-form-search input[name="posted_at_instrument"]').change( function() {
			$('#el-s-ms-info').html('Updating tags and authors...');
			var params = {id: that.exp_id};
			if($('#elog-form-search input[name="posted_at_instrument"]').attr('checked')) params.accross_instrument = 1;
			var jqXHR = $.get('../logbook/RequestUsedTagsAndAuthors.php',params,function(data) {
				var result = eval(data);
				if(result.Status != 'success') {
					$('#el-s-ms-info').html(result.Message);
					return;
				}
				$('#el-s-ms-info').html('');
				var tags_html = '<option></option>';
				var tags = result.Tags;
				for(var i=0; i < tags.length; i++) tags_html += '<option>'+tags[i]+'</option>';
				$('#elog-form-search select[name="tag"]').html(tags_html);
				var authors_html = '<option></option>';
				var authors = result.Authors;
				for(var i=0; i < authors.length; i++) authors_html += '<option>'+authors[i]+'</option>';
				$('#elog-form-search select[name="author"]').html(authors_html);
			},
			'JSON').error(function () {	that.report_error( 'failed because of: '+jqXHR.statusText ); });
		});
		$('#elog-form-search').find('input[name="runs"]').keyup(function(e) { if( e.keyCode == 13 ) that.search(); });
	};
	this.simple_search = function(text2search) {
		this.search_reset();
		$('#elog-form-search input[name="text2search"]').val(text2search);
		var search_in_messages   = 1,
		    search_in_tags       = 0,
		    search_in_values     = 0,
		    posted_at_instrument = 0,
		    posted_at_experiment = 1,
		    posted_at_shifts     = 1,
		    posted_at_runs       = 1,
		    begin                = '',
		    end                  = '',
		    tag                  = '',
		    author               = '',
		    range_of_runs        = '';
		this.search_message_viewer.search(
			text2search,
			search_in_messages,
			search_in_tags,
			search_in_values,
			posted_at_instrument,
			posted_at_experiment,
			posted_at_shifts,
			posted_at_runs,
			begin,
			end,
			tag,
			author,
			range_of_runs
		);
	};
	this.search = function() {
		function is_checked(checkbox) {
			return $('#elog-form-search input[name="'+checkbox+'"]').attr('checked') ? 1 : 0;
		}
		this.search_message_viewer.search(
			$('#elog-form-search input[name="text2search"]').val(),
			is_checked('search_in_messages'),
			is_checked('search_in_tags'),
			is_checked('search_in_values'),
			is_checked('posted_at_instrument'),
			is_checked('posted_at_experiment'),
			is_checked('posted_at_shifts'),
			is_checked('posted_at_runs'),
			$('#elog-form-search input[name="begin"]').val(),
			$('#elog-form-search input[name="end"]').val(),
			$('#elog-form-search select[name="tag"]').val(),
			$('#elog-form-search select[name="author"]').val(),
			$('#elog-form-search input[name="runs"]').val()
		);
	};

	/* --------------------------------------------
	 *  Reset the search form to its initial state
	 * --------------------------------------------
	 */
	this.search_reset = function() {
		function set_checked(checkbox, on) {
			if(on) $('#elog-form-search input[name="'+checkbox+'"]').attr('checked','checked');
			else $('#elog-form-search input[name="'+checkbox+'"]').removeAttr('checked');
		}
		$('#elog-form-search input[name="text2search"]').val('');
		set_checked('search_in_messages',   1);
		set_checked('search_in_tags',       0);
		set_checked('search_in_values',     0);
		set_checked('posted_at_instrument', 0);
		set_checked('posted_at_experiment', 1);
		set_checked('posted_at_shifts',     1);
		set_checked('posted_at_runs',       1);
		$('#elog-form-search input[name="begin"]').val('');
		$('#elog-form-search input[name="end"]').val('');
		$('#elog-form-search select[name="tag"]').val('');
		$('#elog-form-search select[name="author"]').val('');
		$('#elog-form-search input[name="runs"]').val('');
	};

	/* ---------------------------------------------
	 *  Initialize the page with the list of shifts
	 * ---------------------------------------------
	 */
	this.shifts_last_request = null;
	this.shifts_max_seconds = 1;
	this.toggle_shift = function(idx) {
		var s = this.shifts_last_request[idx];
		var toggler='#el-sh-tgl-'+s.id;
		var container='#el-sh-con-'+s.id;
		if( $(container).hasClass('el-sh-hdn')) {
			$(toggler).removeClass('ui-icon-triangle-1-e').addClass('ui-icon-triangle-1-s');
			$(container).removeClass('el-sh-hdn').addClass('el-sh-vis');
		} else {
			$(toggler).removeClass('ui-icon-triangle-1-s').addClass('ui-icon-triangle-1-e');
			$(container).removeClass('el-sh-vis').addClass('el-sh-hdn');
		}

	};
	this.shift2html = function(idx) {
		var s = this.shifts_last_request[idx];
		var crew = '';
		for( var i=0; i < s.crew.length; i++ ) {
			crew += s.crew[i]+'<br>';
		}
		var runs = '';
		for( var i=0; i < s.runs.length; i++ ) {
			var r = s.runs[i];
			runs += '<div style="float:left; margin-right:10px;"><a class="link" href="../logbook/?action=select_run_by_id&id='+r.id+'" target="_blank">'+r.num+'</a></div>';
		}
		if( runs != '' ) runs += '<div style="clear:both;"></div>';
		var html =
'<div class="el-sh-hdr" id="el-sh-hdr-'+idx+'" onclick="elog.toggle_shift('+idx+');">'+
'  <div style="float:left; margin-right:20px;"><span class="toggler ui-icon ui-icon-triangle-1-e el-sh-tgl" id="el-sh-tgl-'+s.id+'"></span></div>'+
'  <div style="float:left;"><b>begin:</b></div>'+
'  <div style="float:left;" class="el-sh-begin">'+s.begin+'</div>'+
'  <div style="float:left;"><b>end:</b></div>'+
'  <div style="float:left;" class="el-sh-end">'+s.end+'</div>'+
'  <div style="float:left;"><b>leader:</b></div>'+
'  <div style="float:left;" class="el-sh-leader">'+s.leader+'</div>'+
'  <div style="float:left;"><b># of runs:</b></div>'+
'  <div style="float:left;" class="el-sh-runs">'+s.num_runs+'</div>'+
'  <div style="float:left;"><b>duration:</b></div>'+
'  <div style="float:left;" class="el-sh-durat">'+s.durat+'</div>'+
'  <div style="float:left; width:'+Math.floor(200*(s.sec/this.shifts_max_seconds))+'px;" class="el-sh-bar">&nbsp;</div>'+
'  <div style="clear:both;"></div>'+
'</div>'+
'<div class="el-sh-con el-sh-hdn" id="el-sh-con-'+s.id+'">'+
'  <table><tbody>'+
'    <tr>'+
'      <td class="table_cell table_cell_left  table_cell_top">Crew:&nbsp;&nbsp;</td>'+
'      <td class="table_cell table_cell_right table_cell_top">'+crew+'</td></tr>'+
'    <tr>'+
'     <td class="table_cell table_cell_left  ">Goals:&nbsp;&nbsp;</td>'+
'     <td class="table_cell table_cell_right "><textbox rows="10" cols="64">'+s.goals+'</textbox ></td></tr>'+
'    <tr>'+
'     <td class="table_cell table_cell_left  table_cell_bottom">Runs:&nbsp;&nbsp;</td>'+
'     <td class="table_cell table_cell_right table_cell_bottom">'+runs+'</td></tr>'+
'  </tbody></table>'+
'</div>';
		return html;
	};
	this.display_shifts = function() {
		var shifts = this.shifts_last_request;
		var html = '';
		for( var i=0; i < shifts.length; i++ ) {
			html += this.shift2html(i);
		}
		$('#el-sh-list').html(html);
	};
	this.sort_shifts = function() {
		function compare_elements_by_begin(a, b) { return b.begin_sec - a.begin_sec; }
		function compare_elements_by_runs(a, b) { return b.num_runs - a.num_runs; }
		function compare_elements_by_duration(a, b) { return b.sec - a.sec; }
		var sort_function = null;
		switch( $('#el-sh-wa' ).find('select[name="sort"]').val()) {
		case 'begin'   : sort_function = compare_elements_by_begin; break;
		case 'runs'    : sort_function = compare_elements_by_runs; break;
		case 'duration': sort_function = compare_elements_by_duration; break;
		}
		this.shifts_last_request.sort( sort_function );
	};
	this.update_shifts = function() {
		$('#el-sh-updated').html('Updating shifts...');
		var params = {exper_id: that.exp_id};
		var jqXHR = $.get('../logbook/RequestAllShifts.php',params,function(data) {
			var result = eval(data);
			if(result.Status != 'success') {
				$('#el-sh-updated').html(result.Message);
				return;
			}
			that.shifts_last_request = result.Shifts;
			that.shifts_max_seconds  = result.MaxSeconds;
			that.sort_shifts();
			that.display_shifts();
			$('#el-sh-info').html('<b>'+that.shifts_last_request.length+'</b> shifts');
			$('#el-sh-updated').html('[ Last update on: <b>'+result.Updated+'</b> ]');
		},
		'JSON').error(function () { that.report_error( 'failed because of: '+jqXHR.statusText ); });
	};
	this.shifts_init = function() {
		$('#el-sh-refresh').button().click(function() { that.update_shifts(); });
		$('#el-sh-wa' ).find('select[name="sort"]').change(function() {
			that.sort_shifts();
			that.display_shifts();
		});
		$('#el-sh-reverse').button().click(function() {
			that.shifts_last_request.reverse();
			that.display_shifts();
		});
		$('#el-sh-new-begin').button().click(function() {
			$('#el-sh-new-wa').removeClass("el-sh-new-hdn").addClass("el-sh-new-vis");
			$(this).button("disable");
		});
		$('#el-sh-new-submit').button().click(function() {
			$('#el-sh-new-wa').removeClass("el-sh-new-vis").addClass("el-sh-new-hdn");
			$('#el-sh-new-begin').button("enable");
			var urlbase = window.location.href;
			var idx = urlbase.indexOf( '&' );
			if( idx > 0 ) urlbase = urlbase.substring( 0, idx );
			$('#elog_new_shift_form input[name="actionSuccess"]').val(urlbase+'&app=elog:shifts');
			$('#elog_new_shift_form').trigger('submit');
		});
		$('#el-sh-new-cancel').button().click(function() {
			$('#el-sh-new-wa').removeClass("el-sh-new-vis").addClass("el-sh-new-hdn");
			$('#el-sh-new-begin').button("enable");
		});
		this.update_shifts();
	};

	/* -------------------------------------------
	 *  Initialize the page with the list of runs
	 * -------------------------------------------
	 */
	this.runs_last_request = null;
	this.runs_max_seconds = 1;
	this.toggle_run = function(idx) {
		var r = this.runs_last_request[idx];
		var toggler='#el-r-tgl-'+r.id;
		var container='#el-r-con-'+r.id;
		if( $(container).hasClass('el-r-hdn')) {
			$(toggler).removeClass('ui-icon-triangle-1-e').addClass('ui-icon-triangle-1-s');
			$(container).removeClass('el-r-hdn').addClass('el-r-vis');
			$('#el-r-con-'+r.id).html('Loading...');

			// Instantiate the simplified message viewer and let it full control over
			// the container for displaying anything which is related to the run.
			//
			r.viewer = new elog_message_viewer4run_create('elog.runs_last_request['+idx+'].viewer', this, 'el-r-con-'+r.id, r.num);
			r.viewer.reload();

		} else {
			$(toggler).removeClass('ui-icon-triangle-1-s').addClass('ui-icon-triangle-1-e');
			$(container).removeClass('el-r-vis').addClass('el-r-hdn');

			// Clear the container and get rid of the viewer, so that
			// we'll be able to reload it fresh next time the container is open.
			//
			r.viewer = null;
			$(container).html('');
		}

	};
	this.run2html = function(idx) {
		var r = this.runs_last_request[idx];
		var html =
'  <div class="el-r-hdr" id="el-r-hdr-'+idx+'" onclick="elog.toggle_run('+idx+');">'+
'    <div style="float:left;"><span class="toggler ui-icon ui-icon-triangle-1-e el-r-tgl" id="el-r-tgl-'+r.id+'"></span></div>'+
'    <div style="float:left;" class="el-r-num">'+r.num+'</div>'+
'    <div style="float:left;" class="el-r-day">'+r.day+'</div>'+
'    <div style="float:left;" class="el-r-ival">'+r.ival+'</div>'+
'    <div style="float:left;"><b>duration:&nbsp;&nbsp;</b></div>'+
'    <div style="float:left;" class="el-r-durat">'+r.durat+'</div>'+
'    <div style="float:left; width:'+Math.floor(200*(r.sec/this.runs_max_seconds))+'px;" class="el-r-bar">&nbsp;</div>'+
'    <div style="clear:both;"></div>'+
'  </div>'+
'  <div class="el-r-con el-r-hdn" id="el-r-con-'+r.id+'"></div>';
		return html;
	};
	this.display_runs = function() {
		var runs = this.runs_last_request;
		var html = '';
		for( var i=0; i < runs.length; i++ ) {
			runs[i].loaded = false;
			html += this.run2html(i);
		}
		$('#el-r-list').html(html);
	};
	this.sort_runs = function() {
		function compare_elements_by_run(a, b) { return b.num - a.num; }
		function compare_elements_by_duration(a, b) { return b.sec - a.sec; }
		var sort_function = null;
		switch( $('#el-r-wa' ).find('select[name="sort"]').val()) {
		case 'run'     : sort_function = compare_elements_by_run; break;
		case 'duration': sort_function = compare_elements_by_duration; break;
		}
		this.runs_last_request.sort( sort_function );
	};
	this.update_runs = function() {
		$('#el-r-updated').html('Updating runs...');
		var params = {exper_id: that.exp_id};
		var jqXHR = $.get('../logbook/RequestAllRuns.php',params,function(data) {
			var result = eval(data);
			if(result.Status != 'success') {
				$('#el-r-updated').html(result.Message);
				return;
			}
			that.runs_last_request = result.Runs;
			that.runs_max_seconds  = result.MaxSeconds;
			that.sort_runs();
			that.display_runs();
			$('#el-r-info').html('<b>'+that.runs_last_request.length+'</b> runs');
			$('#el-r-updated').html('[ Last update on: <b>'+result.Updated+'</b> ]');
		},
		'JSON').error(function () { that.report_error( 'failed because of: '+jqXHR.statusText ); });
	};
	this.runs_init = function() {
		$('#el-r-refresh').button().click(function() { that.update_runs(); });
		$('#el-r-wa' ).find('select[name="sort"]').change(function() {
			that.sort_runs();
			that.display_runs();
		});
		$('#el-r-reverse').button().click(function() {
			that.runs_last_request.reverse();
			that.display_runs();
		});
		this.update_runs();
	};

	/* --------------------------------------------------
	 *  Initialize the page with the list of attachments
	 * --------------------------------------------------
	 */
	this.attachments_last_request = null;
	this.sort_attachments = function() {
		function compare_elements_by_posted(a, b) { return b.time64 - a.time64; }
		function compare_elements_by_author(a, b) { if( a.type != 'a' ) return -1; if( b.type != 'a' ) return 1; return ( a.e_author  < b.e_author ? -1 : (a.e_author > b.e_author ? 1 : 0 )); }
		function compare_elements_by_name  (a, b) { if( a.type != 'a' ) return -1; if( b.type != 'a' ) return 1; return ( a.a_name    < b.a_name   ? -1 : (a.a_name   > b.a_name   ? 1 : 0 )); }
		function compare_elements_by_type  (a, b) { if( a.type != 'a' ) return -1; if( b.type != 'a' ) return 1; return ( a.a_type    < b.a_type   ? -1 : (a.a_type   > b.a_type   ? 1 : 0 )); }
		function compare_elements_by_size  (a, b) { if( a.type != 'a' ) return -1; if( b.type != 'a' ) return 1; return   b.a_size    - a.a_size; }
		var sort_function = null;
		switch( $('#el-at-wa' ).find('select[name="sort"]').val()) {
		case 'posted': sort_function = compare_elements_by_posted; break;
		case 'author': sort_function = compare_elements_by_author; break;
		case 'name':   sort_function = compare_elements_by_name;   break;
		case 'type':   sort_function = compare_elements_by_type;   break;
		case 'size':   sort_function = compare_elements_by_size;   break;
		}
		this.attachments_last_request.sort( sort_function );
	};
	this.display_attachments_as_table = function() {
		var html =
'<table><tbody>'+
'  <tr>'+
'    <td class="table_hdr">Host Message</td>'+
'    <td class="table_hdr">Posted</td>'+
'    <td class="table_hdr">Author</td>'+
'    <td class="table_hdr">Attachment Name</td>'+
'    <td class="table_hdr">Type</td>'+
'    <td class="table_hdr">Size</td>'+
'  </tr>';
		var attachments = this.attachments_last_request;
		for(var i=0; i < attachments.length; i++) {
			var extra_class = (i == attachments.length-1 ? 'table_cell_bottom' : '');
			var a = attachments[i];
			if( a.type != 'a' ) continue;
			html +=
'  <tr>'+
'    <td class="table_cell '+extra_class+' table_cell_left">' +a.e_url   +'</td>'+
'    <td class="table_cell '+extra_class+'">'                 +a.e_time  +'</td>'+
'    <td class="table_cell '+extra_class+'">'                 +a.e_author+'</td>'+
'    <td class="table_cell '+extra_class+'">'                 +a.a_url   +'</td>'+
'    <td class="table_cell '+extra_class+'">'                 +a.a_type  +'</td>'+
'    <td class="table_cell '+extra_class+' table_cell_right">'+a.a_size  +'</td>'+
'  </tr>';
		}
		html +=
'</tbody></table>';
		return html;
	};
	this.display_attachments_as_table_with_runs = function() {
		var html =
'<table><tbody>'+
'  <tr>'+
'    <td class="table_hdr">Run</td>'+
'    <td class="table_hdr">Started</td>'+
'    <td class="table_hdr">Message</td>'+
'    <td class="table_hdr">Posted</td>'+
'    <td class="table_hdr">Author</td>'+
'    <td class="table_hdr">Attachment Name</td>'+
'    <td class="table_hdr">Type</td>'+
'    <td class="table_hdr">Size</td>'+
'  </tr>';
		var run_specific_style = 'style="background-color:#f0f0f0;"';
		var attachments = this.attachments_last_request;
		for(var i=0; i < attachments.length; i++) {
			var extra_class = (i == attachments.length-1 ? 'table_cell_bottom' : '');
			var a = attachments[i];
			if( a.type == 'a' ) {
				html +=
'  <tr>'+
'    <td class="table_cell '+extra_class+' table_cell_left">'            +'</td>'+
'    <td class="table_cell '+extra_class+'">'                            +'</td>'+
'    <td class="table_cell '+extra_class+'">'                 +a.e_url   +'</td>'+
'    <td class="table_cell '+extra_class+'">'                 +a.e_time  +'</td>'+
'    <td class="table_cell '+extra_class+'">'                 +a.e_author+'</td>'+
'    <td class="table_cell '+extra_class+'">'                 +a.a_url   +'</td>'+
'    <td class="table_cell '+extra_class+'">'                 +a.a_type  +'</td>'+
'    <td class="table_cell '+extra_class+' table_cell_right">'+a.a_size  +'</td>'+
'  </tr>';
			} else {
				html +=
'  <tr>'+
'    <td class="table_cell '+extra_class+' table_cell_left" ' +run_specific_style+'>'+a.r_url  +'</td>'+
'    <td class="table_cell '+extra_class+'" '                 +run_specific_style+'>'+a.r_begin+'</td>'+
'    <td class="table_cell '+extra_class+'" '                 +run_specific_style+'>'          +'</td>'+
'    <td class="table_cell '+extra_class+'" '                 +run_specific_style+'>'          +'</td>'+
'    <td class="table_cell '+extra_class+'" '                 +run_specific_style+'>'          +'</td>'+
'    <td class="table_cell '+extra_class+'" '                 +run_specific_style+'>'          +'</td>'+
'    <td class="table_cell '+extra_class+'" '                 +run_specific_style+'>'          +'</td>'+
'    <td class="table_cell '+extra_class+' table_cell_right" '+run_specific_style+'>'          +'</td>'+
'  </tr>';
			}
		}
		html +=
'</tbody></table>';
		return html;
	};
	this.display_attachments_as_thumbnail = function() {
		var html = '';
		var attachments = this.attachments_last_request;
		for(var i=0; i < attachments.length; i++) {
			var extra_class = (i == attachments.length-1 ? 'table_cell_bottom' : '');
			var a = attachments[i];
			if( a.type != 'a' ) continue;
			var title =
				'name: '+a.a_name+'\n'+
				'type: '+a.a_type+'\n'+
				'size: '+a.a_size+'\n'+
				'posted: '+a.e_time+'\n'+
				'author: '+a.e_author;
			html +=
'<div style="float:left;" title="'+title+'"><a href="../logbook/attachments/'+a.a_id+'/'+a.a_name+'" target="_blank"><img style="height:160px; padding:8px;" src="../logbook/attachments/preview/'+a.a_id+'" /></a></div>';
		}
		return html;
	};
	this.display_attachments_as_thumbnail_with_runs = function() {
		var html = '';
		var attachments = this.attachments_last_request;
		for(var i=0; i < attachments.length; i++) {
			var extra_class = (i == attachments.length-1 ? 'table_cell_bottom' : '');
			var a = attachments[i];
			if( a.type == 'a' ) {
				var title =
					'name: '+a.a_name+'\n'+
					'type: '+a.a_type+'\n'+
					'size: '+a.a_size+'\n'+
					'posted: '+a.e_time+'\n'+
					'author: '+a.e_author;
				html +=
'<div style="float:left;" title="'+title+'"><a href="../logbook/attachments/'+a.a_id+'/'+a.a_name+'" target="_blank"><img style="height:160px;  border-top:solid 1px #d0d0d0; padding:8px; padding-left:0px;" src="../logbook/attachments/preview/'+a.a_id+'" /></a></div>';
			} else {
				var title =
					'run #: '+a.r_num+'\n'+
					'begin: '+a.r_begin+'\n'+
					'end: '+a.r_end;
				html +=
'<div style="float:left; height:160px; padding:8px; border-left:solid 1px #d0d0d0; border-top:solid 1px #d0d0d0; font-weight:bold;" title="'+title+'">'+a.r_num+' . .</div>';
			}
		}
		return html;
	};
	this.display_attachments_as_hybrid = function() {
		var html =
'<table><tbody>'+
'  <tr>'+
'    <td class="table_hdr">Attachment</td>'+
'    <td class="table_hdr">Info</td>'+
'  </tr>';
		var title = 'open the attachment in a separate tab';
		var attachments = this.attachments_last_request;
		for(var i=0; i < attachments.length; i++) {
			var extra_class = (i == attachments.length-1 ? 'table_cell_bottom' : '');
			var a = attachments[i];
			if( a.type != 'a' ) continue;
			var thumb =
'<div style="float:left;" title="'+title+'">'+
'  <a href="../logbook/attachments/'+a.a_id+'/'+a.a_name+'" target="_blank">'+
'    <img style="height:130px;" src="../logbook/attachments/preview/'+a.a_id+'"/>'+
'  </a>'+
'</div>';
			var info =
'<table><tbody>'+
'  <tr>'+
'    <td class="table_cell table_cell_top table_cell_left">Host Message</td>'+
'    <td class="table_cell table_cell_top table_cell_right">'+a.e_url+'</td>'+
'  </tr>'+
'  <tr>'+
'    <td class="table_cell table_cell_left">Posted</td>'+
'    <td class="table_cell table_cell_right">'+a.e_time+'</td>'+
'  </tr>'+
'  <tr>'+
'    <td class="table_cell table_cell_left">Author</td>'+
'    <td class="table_cell table_cell_right">'+a.e_author+'</td>'+
'  </tr>'+
'  <tr>'+
'    <td class="table_cell table_cell_left">Attachment Name</td>'+
'    <td class="table_cell table_cell_right">'+a.a_url+'</td>'+
'  </tr>'+
'  <tr>'+
'    <td class="table_cell table_cell_left">Type</td>'+
'    <td class="table_cell table_cell_right">'+a.a_type+'</td>'+
'  </tr>'+
'  <tr>'+
'    <td class="table_cell table_cell_bottom table_cell_left">Size</td>'+
'    <td class="table_cell table_cell_bottom table_cell_right">'+a.a_size+'</td>'+
'  </tr>'+
'</tbody></table>';
			html +=
'  <tr>'+
'    <td class="table_cell '+extra_class+' table_cell_left">' +thumb+'</td>'+
'    <td class="table_cell '+extra_class+' table_cell_right">'+info +'</td>'+
'  </tr>';
		}
		html +=
'</tbody></table>';
		return html;
	};
	this.display_attachments_as_hybrid_with_runs = function() {
		var run_specific_style = 'style="background-color:#f0f0f0;"';
		var html =
'<table><tbody>'+
'  <tr>'+
'    <td class="table_hdr">Run</td>'+
'    <td class="table_hdr">Started</td>'+
'    <td class="table_hdr">Attachment</td>'+
'    <td class="table_hdr">Info</td>'+
'  </tr>';
		var title = 'open the attachment in a separate tab';
		var attachments = this.attachments_last_request;
		for(var i=0; i < attachments.length; i++) {
			var extra_class = (i == attachments.length-1 ? 'table_cell_bottom' : '');
			var a = attachments[i];
			if( a.type == 'a' ) {
				var thumb =
'<div style="float:left;" title="'+title+'">'+
'  <a href="../logbook/attachments/'+a.a_id+'/'+a.a_name+'" target="_blank">'+
'    <img style="height:130px;" src="../logbook/attachments/preview/'+a.a_id+'"/>'+
'  </a>'+
'</div>';
				var info =
'<table><tbody>'+
'  <tr>'+
'    <td class="table_cell table_cell_top table_cell_left">Host Message</td>'+
'    <td class="table_cell table_cell_top table_cell_right">'+a.e_url+'</td>'+
'  </tr>'+
'  <tr>'+
'    <td class="table_cell table_cell_left">Posted</td>'+
'    <td class="table_cell table_cell_right">'+a.e_time+'</td>'+
'  </tr>'+
'  <tr>'+
'    <td class="table_cell table_cell_left">Author</td>'+
'    <td class="table_cell table_cell_right">'+a.e_author+'</td>'+
'  </tr>'+
'  <tr>'+
'    <td class="table_cell table_cell_left">Attachment Name</td>'+
'    <td class="table_cell table_cell_right">'+a.a_url+'</td>'+
'  </tr>'+
'  <tr>'+
'    <td class="table_cell table_cell_left">Type</td>'+
'    <td class="table_cell table_cell_right">'+a.a_type+'</td>'+
'  </tr>'+
'  <tr>'+
'    <td class="table_cell table_cell_bottom table_cell_left">Size</td>'+
'    <td class="table_cell table_cell_bottom table_cell_right">'+a.a_size+'</td>'+
'  </tr>'+
'</tbody></table>';
				html +=
'  <tr>'+
'    <td class="table_cell '+extra_class+' table_cell_left"></td>'+
'    <td class="table_cell '+extra_class+'"></td>'+
'    <td class="table_cell '+extra_class+'">' +thumb+'</td>'+
'    <td class="table_cell '+extra_class+' table_cell_right">'+info +'</td>'+
'  </tr>';
			} else {
				html +=
'  <tr>'+
'    <td class="table_cell '+extra_class+' table_cell_left" ' +run_specific_style+'>'+a.r_url  +'</td>'+
'    <td class="table_cell '+extra_class+'" '                 +run_specific_style+'>'+a.r_begin+'</td>'+
'    <td class="table_cell '+extra_class+'" '                 +run_specific_style+'>'          +'</td>'+
'    <td class="table_cell '+extra_class+' table_cell_right" '+run_specific_style+'>'          +'</td>'+
'  </tr>';
			}
		}
		html +=
'</tbody></table>';
		return html;
	};
	this.display_attachments = function() {
		var html='';
		var display_name =
			$('#el-at-wa' ).find('select[name="view"]').val()+
			($('#el-at-wa' ).find('select[name="runs"]').val() == 'yes' ? 'runs' : '');
		switch(display_name) {
		case 'table'         : html = this.display_attachments_as_table              (); break;
		case 'tableruns'     : html = this.display_attachments_as_table_with_runs    (); break;
		case 'thumbnails'    : html = this.display_attachments_as_thumbnail          (); break;
		case 'thumbnailsruns': html = this.display_attachments_as_thumbnail_with_runs(); break;
		case 'hybrid'        : html = this.display_attachments_as_hybrid             (); break;
		case 'hybridruns'    : html = this.display_attachments_as_hybrid_with_runs   (); break;
		}
		$('#el-at-list').html(html);
	};
	this.update_attachments = function() {
		$('#el-at-updated').html('Updating attachments...');
		var params = {exper_id: that.exp_id};
		var jqXHR = $.get('../logbook/RequestAllAttachments.php',params,function(data) {
			var result = eval(data);
			if(result.Status != 'success') {
				$('#el-at-updated').html(result.Message);
				return;
			}
			that.attachments_last_request = result.Attachments;
			that.sort_attachments();
			that.display_attachments();
			var num_attachments = 0;
			for(var i=0; i < that.attachments_last_request.length; i++) {
				var a = that.attachments_last_request[i];
				if( a.type == 'a' ) num_attachments++;
			}
			$('#el-at-info').html('<b>'+num_attachments+'</b> attachments');
			$('#el-at-updated').html('[ Last update on: <b>'+result.Updated+'</b> ]');
		},
		'JSON').error(function () { that.report_error( 'failed because of: '+jqXHR.statusText ); });
	};
	this.attachments_init = function() {
		$('#el-at-refresh').button().click(function() { that.update_attachments(); });
		$('#el-at-wa').find('select[name="sort"]').change(function(){
			that.sort_attachments();
			that.display_attachments();
		});
		$('#el-at-wa').find('select[name="view"]').change(function(){
			that.display_attachments();
		});
		$('#el-at-wa').find('select[name="runs"]').change(function(){
			that.display_attachments();
		});
		$('#el-at-reverse').button().click(function() {
			that.attachments_last_request.reverse();
			that.display_attachments();
		});
		this.update_attachments();
	};

	/* ----------------------------------------------
	 *  Initialize page with subscription management
	 * ----------------------------------------------
	 */
	this.subscription = function(operation, id) {
		var params = {exper_id: this.exp_id, operation: operation};
		if( id != null ) params.id = id;
		var jqXHR = $.get('../logbook/CheckSubscription.php',params,function(data) {
			var result = eval(data);
			if(result.Status != 'success') { that.report_error( result.Message ); return; }
			$('#el-subscribed'  ).css('display', result.Subscribed ? 'block' : 'none' );
			$('#el-unsubscribed').css('display', result.Subscribed ? 'none'  : 'block');
			var html = '';
			var all_subscriptions = result.AllSubscriptions;
			for( var i=0; i < all_subscriptions.length; i++) {
				var s = all_subscriptions[i];
				var extra_class = (i == all_subscriptions.length-1 ? 'table_cell_bottom' : '');
				html +=
'  <tr><td class="table_cell '+extra_class+' table_cell_left" >'+s.address        +'</td>'+
'      <td class="table_cell '+extra_class+' "                >'+s.subscriber     +'</td>'+
'      <td class="table_cell '+extra_class+' "                >'+s.subscribed_host+'</td>'+
'      <td class="table_cell '+extra_class+' "                >'+s.subscribed_time+'</td>'+
'      <td class="table_cell '+extra_class+' table_cell_right"><button title="Stop receiving automated notifications" value='+s.id+'>Unsubscribe</button></td></tr>';
			}
			if( html != '' ) {
				html =
'<h3>All subscriptions for this e-Log:</h3>'+
'<table style="padding-left:10px;"><tbody>'+
'  <tr><td class="table_hdr">Recipient</td>'+
'      <td class="table_hdr">Subscribed by</td>'+
'      <td class="table_hdr">From host</td>'+
'      <td class="table_hdr">Date</td>'+
'      <td class="table_hdr">Actions</td></tr>'+
html+
'</tbody></table>';

			}
			$('#el-subscribe-all').html(html);
			$('#el-subscribe-all').find('button').button().click(function() {
				that.subscription('UNSUBSCRIBE', $(this).val());
			});
		},
		'JSON').error(function () { that.report_error( 'failed because of: '+jqXHR.statusText ); });
	};
	this.subscribe_init = function() {
		$('#el-subscribe'  ).button().click(function() { that.subscription('SUBSCRIBE',   null); });
		$('#el-unsubscribe').button().click(function() { that.subscription('UNSUBSCRIBE', null); });
		this.subscription('CHECK', null);
	};

	/* ----------------------------------
	 *  Application initialization point
	 * ----------------------------------
	 *
	 * RETURN: true if the real initialization took place. Return false
	 *         otherwise.
	 */
	this.is_initialized = false;
	this.init = function() {
		if(this.is_initialized) return false;
		this.is_initialized = true;
		this.live_init();
		this.post_init();
		this.search_init();
		this.shifts_init();
		this.runs_init();
		this.attachments_init();
		this.subscribe_init();
		return true;
	};
}

var elog = new elog_create();

/**
 * An object for displaying e-log messages.
 * 
 * @object_address - the full path name to an instance of the object (used for global references from collback functions
 * @param parent - parent object
 * @param element_base - the base of an element where to grow the DOM
 * @return
 */
function elog_message_viewer_create(object_address, parent_object, element_base) {

	this.address = object_address;
	this.parent = parent_object;
	this.base = element_base;

	this.reverse_order = false;

	this.last_updated = '';
	this.expand_collapse = 0;
	this.days2threads = null;
	this.current_day = null;
	this.threads = null;
	this.messages = null;
	this.total_messages = 0;
	this.runs = null;
	this.min_run = 0;
	this.max_run = 0;
	this.attachments = null;
	this.attachments_loader = null;

	var that = this;

	$('#'+this.base+'-expand').button().click(function() {
		switch( that.expand_collapse ) {
		default:
		case 1:
			that.expand_all_messages();
		case 0:
			$('.el-l-m-d-tgl').removeClass('ui-icon-triangle-1-e').addClass('ui-icon-triangle-1-s');
			$('.el-l-m-d-con').removeClass('el-l-m-d-hdn').addClass('el-l-m-d-vis');
		}
		that.expand_collapse++;
		if( that.expand_collapse > 2 ) that.expand_collapse = 2;
	});
	$('#'+this.base+'-collapse').button().click(function() {
		that.dim_day();
		switch( that.expand_collapse ) {
		default:
		case 1:
			$('.el-l-m-d-tgl').removeClass('ui-icon-triangle-1-s').addClass('ui-icon-triangle-1-e');
			$('.el-l-m-d-con').removeClass('el-l-m-d-vis').addClass('el-l-m-d-hdn');
		case 2:
			$('.el-l-m-tgl').removeClass('ui-icon-triangle-1-s').addClass('ui-icon-triangle-1-e');
			$('.el-l-m-con').removeClass('el-l-m-vis').addClass('el-l-m-hdn');
			that.collapse_all_runs();
		}
		that.expand_collapse--;
		if( that.expand_collapse < 0 ) that.expand_collapse = 0;
	});
	$('#'+this.base+'-viewattach').button().click(function() {
		that.dim_day();
		$('.el-l-a-tgl').removeClass('ui-icon-triangle-1-e').addClass('ui-icon-triangle-1-s');
		$('.el-l-a-con').removeClass('el-l-a-hdn').addClass('el-l-a-vis');
	});
	$('#'+this.base+'-hideattach').button().click(function() {
		that.dim_day();
		$('.el-l-a-tgl').removeClass('ui-icon-triangle-1-s').addClass('ui-icon-triangle-1-e');
		$('.el-l-a-con').removeClass('el-l-a-vis').addClass('el-l-a-hdn');
	});
	$('#'+this.base+'-reverse').button().click(function() {
		that.dim_day();
		that.days2threads.reverse();
		for(var day_idx = that.days2threads.length-1; day_idx >= 0; day_idx--)
			that.days2threads[day_idx].threads.reverse();
		that.reverse_order = !that.reverse_order;
		that.redisplay();
	});

	this.expand_group_day = function(idx, on) {
		var toggler='#'+that.base+'-m-d-tgl-'+idx;
		var container='#'+that.base+'-m-d-con-'+idx;
		if(on) {
			that.highlight_day(idx);
			$(toggler).removeClass('ui-icon-triangle-1-e').addClass('ui-icon-triangle-1-s');
			$(container).removeClass('el-l-m-d-hdn').addClass('el-l-m-d-vis');
		} else {
			that.dim_day();
			$(toggler).removeClass('ui-icon-triangle-1-s').addClass('ui-icon-triangle-1-e');
			$(container).removeClass('el-l-m-d-vis').addClass('el-l-m-d-hdn');
		}
	};

	this.toggle_group_day = function(idx) {
		var container='#'+that.base+'-m-d-con-'+idx;
		that.expand_group_day(idx, $(container).hasClass('el-l-m-d-hdn'));
	};

	this.expand_message = function(idx, on) {
		var entry = that.threads[idx];
		var toggler='#'+that.base+'-m-tgl-'+entry.id;
		var container='#'+that.base+'-m-con-'+entry.id;

		// REIMPLEMENTED THIS:
		//   Initialize the thread container if this is the first call
		//   to the function for for the message.
		//
		//   if( $(container).html() == '' ) {
		//
		// TO THIS:
		//    Always recreate the thread when opening it to make sure the new contents
		//    gets properly displayed.
		//
		// TODO: Do the same for the message viewer in runs.
		//
		if(on) {
			entry.thread_idx = idx;
			var html =
'    <div class="el-l-m-body">';
			if(this.parent.editor)
				html +=
'      <div style="float:right;" class="s-b-con"><button class="el-l-m-mv"  id="'+this.base+'-m-mv-'+entry.id+'"  onclick="'+this.address+'.live_message_move('+entry.id+',false);">move</button></div>'+
'      <div style="float:right;" class="s-b-con"><button class="el-l-m-ed"  id="'+this.base+'-m-ed-'+entry.id+'"  onclick="'+this.address+'.live_message_edit('+entry.id+',false);">edit</button></div>';
			html +=
'      <div style="float:right; margin-left:5px;" class="s-b-con"><button class="el-l-m-xt" id="'+this.base+'-m-xt-'+entry.id+'" onclick="'+this.address+'.live_message_extend('+entry.id+',false, true);" title="add more tags to the message">+ tags</button></div>'+
'      <div style="float:right; margin-left:5px;" class="s-b-con"><button class="el-l-m-xa" id="'+this.base+'-m-xa-'+entry.id+'" onclick="'+this.address+'.live_message_extend('+entry.id+',false, false);" title="add more attachments to the message">+ attachments</button></div>'+
'      <div style="float:right;"                  class="s-b-con"><button class="el-l-m-re" id="'+this.base+'-m-re-'+entry.id+'" onclick="'+this.address+'.live_message_reply('+entry.id+',false);" title="reply to the message">reply</button></div>'+
'      <div style="float:left; font-size:12px; width:100%; overflow:auto;">'+entry.html1+'</div>'+
'      <div style="clear:both;"></div>'+
'      <div id="'+this.base+'-m-dlgs-'+entry.id+'"></div>'+
'    </div>';
			that.attachments[entry.id] = entry.attachments;
			var attachments_html = '';
			for(var k in entry.attachments) {
				var a = entry.attachments[k];
				that.attachments_loader[a.id] = {loaded: false, descr: a};
				attachments_html +=
'      <div style="float:left;" class="el-l-a">'+
'        <div style="float:left;">'+
'          <span class="toggler ui-icon ui-icon-triangle-1-e el-l-a-tgl" id="'+this.base+'-a-tgl-'+a.id+'" onclick="'+this.address+'.toggle_attachment('+a.id+');"></span>'+
'        </div>'+
'        <div style="float:left;" class="el-l-a-dsc"><a class="link" href="../logbook/attachments/'+a.id+'/'+a.description+'"  target="_blank">'+a.description+'</a></div>'+
'        <div style="float:left; margin-left:10px;" class="el-l-a-info">( type: <b>'+a.type+'</b> size: <b>'+a.size+'</b> )</div>'+
'        <div style="clear:both;"></div>'+
'        <div class="el-l-a-con el-l-a-hdn" id="'+this.base+'-a-con-'+a.id+'">'+
'        </div>'+
'      </div>';
			}
			if(attachments_html) html +=
'    <div class="el-l-m-as">'+attachments_html+
'      <div style="clear:both;"></div>'+
'    </div>';
			var tags_html = '';
			for(var k in entry.tags) {
				if(tags_html) tags_html += ', ';
				tags_html += entry.tags[k].tag;
			}
			if(tags_html) html +=
'    <div class="el-l-m-tags">'+
'      <b><i>keywords: </i></b>'+tags_html+
'    </div>';
			html +=
'    <div id="'+this.base+'-m-c-'+entry.id+'">';
			for(var k in entry.children) {
				var child = entry.children[k];
				if( typeof child == 'string' ) html += that.live_child2html(eval("("+child+")"), idx);
				else                           html += that.live_child2html(child, idx);
			}
			html +=
'    </div>'+
'  </div>';
			$(container).html(html);

			$(container).find('.el-l-m-xt').button();
			$(container).find('.el-l-m-xa').button();
			$(container).find('.el-l-m-re').button();
			$(container).find('.el-l-m-ed').button();
			$(container).find('.el-l-m-mv').button();

			$(toggler).removeClass('ui-icon-triangle-1-e').addClass('ui-icon-triangle-1-s');
			$(container).removeClass('el-l-m-hdn').addClass('el-l-m-vis');
			for(var i = 0; i < entry.attachments.length; i++) {
				this.expand_attachment(entry.attachments[i].id, true);
			}
			for(var i = 0; i < entry.children.length; i++) {
				var child = entry.children[i];
				var child_entry = ( typeof child == 'string' ) ? eval( "("+child+")" ) : child;
				this.expand_child(child_entry.id, true);
			}

		} else {
			$(container).html('');
			$(toggler).removeClass('ui-icon-triangle-1-s').addClass('ui-icon-triangle-1-e');
			$(container).removeClass('el-l-m-vis').addClass('el-l-m-hdn');
		}
	};

	this.expand_all_messages = function() {
		for(var i = that.threads.length-1; i >= 0; i--)
			this.expand_message(i, true);
	};

	this.toggle_message = function(idx) {
		var entry = that.threads[idx];
		var container='#'+that.base+'-m-con-'+entry.id;
		this.expand_message(idx, $(container).hasClass('el-l-m-hdn'));
	};

	this.expand_child = function(id, on) {
		var entry = that.messages[id];
		var toggler='#'+this.base+'-c-tgl-'+id;
		var container='#'+this.base+'-c-con-'+id;
		if(on) {
			$(toggler).removeClass('ui-icon-triangle-1-e').addClass('ui-icon-triangle-1-s');
			$(container).removeClass('el-l-c-hdn').addClass('el-l-c-vis');
			for(var i = 0; i < entry.attachments.length; i++) {
				this.expand_attachment(entry.attachments[i].id, true);
			}
			for(var i = 0; i < entry.children.length; i++) {
				var child = eval( "("+entry.children[i]+")" );
				this.expand_child(child.id, true);
			}
		} else {
			$(toggler).removeClass('ui-icon-triangle-1-s').addClass('ui-icon-triangle-1-e');
			$(container).removeClass('el-l-c-vis').addClass('el-l-c-hdn');
		}
	};

	this.toggle_child = function(id) {
		var container='#'+this.base+'-c-con-'+id;
		this.expand_child(id, $(container).hasClass('el-l-c-hdn'));
	};

	this.expand_attachment = function(id, on) {
		var toggler='#'+this.base+'-a-tgl-'+id;
		var container='#'+this.base+'-a-con-'+id;
		if(on) {
			var a = that.attachments_loader[id];
			if(!a.loaded) {
				a.loaded = true;
			    var html = '<a href="../logbook/attachments/'+id+'/'+a.descr.description+'" target="_blank"><img src="../logbook/attachments/preview/'+id+'" /></a>';
				$(container).html(html);
			}
			$(toggler).removeClass('ui-icon-triangle-1-e').addClass('ui-icon-triangle-1-s');
			$(container).removeClass('el-l-a-hdn').addClass('el-l-a-vis');
		} else {
			$(toggler).removeClass('ui-icon-triangle-1-s').addClass('ui-icon-triangle-1-e');
			$(container).removeClass('el-l-a-vis').addClass('el-l-a-hdn');
		}
	}

	this.toggle_attachment = function(id) {
		var container='#'+this.base+'-a-con-'+id;
		this.expand_attachment(id, $(container).hasClass('el-l-a-hdn'));
	};

	this.toggle_run = function(idx) {
		var entry=this.threads[idx];
		var toggler='#'+this.base+'-r-tgl-'+entry.id;
		var container='#'+this.base+'-r-con-'+entry.id;
		if( $(container).hasClass('el-l-r-hdn')) {
			$(toggler).removeClass('ui-icon-triangle-1-e').addClass('ui-icon-triangle-1-s');
			$(container).removeClass('el-l-r-hdn').addClass('el-l-r-vis');
			if(entry.loaded) return;
			$('#'+this.base+'-r-con-'+entry.id).html('Loading...');
			$.get('../logbook/DisplayRunParams.php',{id:entry.run_id},function(data) {
				var html =
'<div style="float:right;" class="s-b-con"><button class="el-l-r-re" id="'+that.base+'-r-re-'+entry.id+'" onclick="'+that.address+'.live_run_reply('+idx+');">reply</button></div>'+
'<div style="clear:both;"></div>'+
'<div id="'+that.base+'-r-dlgs-'+entry.id+'"></div>'+
'<div style="width:800px; height:300px; overflow:auto; background-color:#ffffff; ">'+data+'</div>';
				$('#'+that.base+'-r-con-'+entry.id).html(html);
				$('#'+that.base+'-r-re-'+entry.id).button();
				entry.loaded = true;
			});
		} else {
			$(toggler).removeClass('ui-icon-triangle-1-s').addClass('ui-icon-triangle-1-e');
			$(container).removeClass('el-l-r-vis').addClass('el-l-r-hdn');
		}
	};
	this.collapse_run = function(idx) {
	   	var entry=this.threads[idx];
		var toggler='#'+this.base+'-r-tgl-'+entry.id;
		var container='#'+this.base+'-r-con-'+entry.id;
		if( $(container).hasClass('el-l-r-vis')) {
			$(toggler).removeClass('ui-icon-triangle-1-s').addClass('ui-icon-triangle-1-e');
			$(container).removeClass('el-l-r-vis').addClass('el-l-r-hdn');
		}
	};

	this.collapse_all_runs = function() {
		for(var i in this.runs) {
			this.collapse_run(this.runs[i]);
		}
	};


	/**
	 * Check if new messages/rusn are available, and if so - refresh the view.
	 * Highlight the new content if the 'highlighter' function is passed as
	 * a parameter.
	 * 
	 * @param inject_runs - check for new run events (begin/end run)
	 * @param highlighter - an optional highlighter function to be called for new messages
	 * @return
	 */
	this.refresh = function(inject_runs, highlighter) {
		if(that.threads.length) {

			var params = {
				id: this.parent.exp_id,
				scope: 'experiment',
				search_in_messages: 1,
				search_in_tags: 1,
				search_in_values: 1,
				posted_at_experiment: 1,
				posted_at_shifts: 1,
				posted_at_runs: 1,
				format: 'detailed',
				since: that.threads[that.threads.length-1].event_timestamp
			};
			if(inject_runs) params.inject_runs = '';

			$.get('../logbook/Search.php',params,function(data) {

				var status = data.ResultSet.Status;
				if(status!='success') {
					$('#'+that.base+'-ms-info').html(data.ResultSet.Message);
					return;
				}

				var new_threads = data.ResultSet.Result;
				if(new_threads.length) {

					var new_days2threads = new Array();

					var last_day = undefined;
					for(var i=0; i < new_threads.length; i++) {
						var entry = new_threads[i];
						var ymd = entry.ymd;
						if((new_days2threads.length == 0) || (last_day != ymd)) {
							new_days2threads.push( { ymd:ymd, runs:0, min_run:0, max_run:0, messages:0, threads: new Array() } );
							last_day = ymd;
						}
						var idx = new_days2threads.length - 1;
						if(entry.is_run) {
							entry.run_num = parseInt(entry.run_num);
							new_days2threads[idx].runs++;
							if((new_days2threads[idx].min_run == 0) || (entry.run_num < new_days2threads[idx].min_run)) new_days2threads[idx].min_run = entry.run_num;
							if((new_days2threads[idx].max_run == 0) || (entry.run_num > new_days2threads[idx].max_run)) new_days2threads[idx].max_run = entry.run_num;
						} else {
							new_days2threads[idx].messages++;
						}
						that.threads.push(entry);
						var thread_idx = that.threads.length-1;
						new_days2threads[idx].threads.push(thread_idx);
					}
					if( that.reverse_order ) {
						new_days2threads.reverse();
						for(var day_idx = new_days2threads.length-1; day_idx >= 0; day_idx--)
							new_days2threads[day_idx].threads.reverse();
					}

					// Merge into ths object's data structures and extend DOM:
					//
					// - if no such day existed before then add the whole day
					// - otherwise merge this day's entries (messages & runs) into existing day
					// - expand messages according to the current state of other messages on the screen
					//
					// ATTENTION: watch for reverse order mode!
					//
					for(var i = new_days2threads.length-1; i >= 0; i--) {
						var found = false;
						for(var day_idx = that.days2threads.length-1; day_idx >= 0; day_idx--) {

							var day = that.days2threads[day_idx];
							if(that.days2threads[day_idx].ymd == new_days2threads[i].ymd) {

								that.days2threads[day_idx].messages += new_days2threads[i].messages;
								that.days2threads[day_idx].runs += new_days2threads[i].runs;

								if(new_days2threads[i].min_run && (new_days2threads[i].min_run < that.days2threads[day_idx].min_run)) that.days2threads[day_idx].min_run = new_days2threads[i].min_run;
								if(new_days2threads[i].max_run && (new_days2threads[i].max_run > that.days2threads[day_idx].max_run)) that.days2threads[day_idx].max_run = new_days2threads[i].max_run;

								for(var j = 0; j < new_days2threads[i].threads.length; j++) {
									var thread_idx = new_days2threads[i].threads[j];
									var html = that.live_thread2html(thread_idx);
									if( that.reverse_order ) {
										that.days2threads[day_idx].threads.unshift(thread_idx);
										$('#'+that.base+'-m-d-con-'+day_idx).append(html);
									} else {
										that.days2threads[day_idx].threads.push(thread_idx);
										$('#'+that.base+'-m-d-con-'+day_idx).prepend(html);
									}
								}
								var runs_messages_info = '<b>'+day.messages+'</b> messages'+( day.runs ? ', runs: <b>'+day.min_run+'</b> .. <b>'+day.max_run+'</b>' : '' );
								$('#'+that.base+'-m-d-con-info-'+day_idx).html(runs_messages_info);

								that.expand_group_day(day_idx, true);
								if(highlighter) highlighter('#'+that.base+'-m-d-hdr-'+day_idx);

								found = true;
								break;
							}
						}
						if(!found) {
							var day_idx = that.days2threads.length-1;
							var html = that.live_day2html(day_idx);
							if( that.reverse_order ) {
								that.days2threads.unshift(new_days2threads[i]);
								$('#'+that.base+'-ms').append(html);
							} else {
								that.days2threads.push(new_days2threads[i]);
								$('#'+that.base+'-ms').prepend(html);
							}
							that.expand_group_day(day_idx, true);
							if(highlighter) highlighter('#'+that.base+'-m-d-hdr-'+day_idx);
						}
						for(var j = 0; j < new_days2threads[i].threads.length; j++) {
							var thread_idx = new_days2threads[i].threads[j];
							if(that.expand_collapse > 1) that.expand_message(thread_idx, true);
							if(highlighter) highlighter('#'+that.base+'-m-hdr-'+thread_idx);
						}
					}
					that.live_update_info();
				}

			},'json');

		} else {
			that.reload(inject_runs, 0);
		}
	};
	this.reload = function(inject_runs, limit) {

		var params = {
			id: this.parent.exp_id,
			scope: 'experiment',
			search_in_messages: 1,
			search_in_tags: 1,
			search_in_values: 1,
			posted_at_experiment: 1,
			posted_at_shifts: 1,
			posted_at_runs: 1,
			format: 'detailed'
		};
		if(inject_runs) params.inject_runs = '';
		if(limit) params.limit = limit;
		
		this.do_reload(params);
	};
	this.search = function(
		text2search,
		search_in_messages,
		search_in_tags,
		search_in_values,
		posted_at_instrument,
		posted_at_experiment,
		posted_at_shifts,
		posted_at_runs,
		begin,
		end,
		tag,
		author,
		range_of_runs) {

		var params = {
			id                  : this.parent.exp_id,
			format              : 'detailed',
			text2search         : text2search,
			search_in_messages  : search_in_messages,
			search_in_tags      : search_in_tags,
			search_in_values    : search_in_values,
			posted_at_instrument: posted_at_instrument,
			posted_at_experiment: posted_at_experiment,
			posted_at_shifts    : posted_at_shifts,
			posted_at_runs      : posted_at_runs,
			begin               : begin,
			end                 : end,
			tag                 : tag,
			author              : author,
			range_of_runs       : range_of_runs
		};
		this.do_reload(params);
	};
	this.do_reload = function(params) {

		that.dim_day();

		$('#'+this.base+'-ms-updated').html('Searching...');
		$.get('../logbook/Search.php',params,function(data) {

			var status = data.ResultSet.Status;
			if(status!='success') {
				$('#'+that.base+'-ms-updated').html(data.ResultSet.Message);
				return;
			}
			$('#'+this.base+'-ms-updated').html('Rendering...');

			that.last_updated = data.ResultSet.Updated;
			that.expand_collapse = 0;
			that.threads = data.ResultSet.Result;
			that.messages = new Array();
			that.total_messages = 0;
			that.runs = new Array();
			that.min_run = 0;
			that.max_run = 0;
			that.attachments = new Array();
			that.attachments_loader = new Array();
			that.days2threads = new Array();

			var last_day = undefined;
			for(var i=0; i < that.threads.length; i++) {
				var entry = that.threads[i];
				var ymd = entry.ymd;
				if((that.days2threads.length == 0) || (last_day != ymd)) {
					that.days2threads.push( { ymd:ymd, runs:0, min_run:0, max_run:0, messages:0, threads: new Array() } );
					last_day = ymd;
				}
				var idx = that.days2threads.length - 1;
				if( entry.is_run ) {
					entry.run_num = parseInt(entry.run_num);
					that.days2threads[idx].runs++;
					if((that.days2threads[idx].min_run == 0) || (entry.run_num < that.days2threads[idx].min_run)) that.days2threads[idx].min_run = entry.run_num;
					if((that.days2threads[idx].max_run == 0) || (entry.run_num > that.days2threads[idx].max_run)) that.days2threads[idx].max_run = entry.run_num;
				} else {
					that.days2threads[idx].messages++;
				}
				that.days2threads[idx].threads.push(i);
			}
			if( that.reverse_order ) {
				that.days2threads.reverse();
				for(var day_idx = that.days2threads.length-1; day_idx >= 0; day_idx--)
					that.days2threads[day_idx].threads.reverse();
			}
/*
			var html = '';
			for(var day_idx = that.days2threads.length-1; day_idx >= 0; day_idx--)
				html += that.live_day2html(day_idx);
			$('#'+that.base+'-ms').html(html);

			if(that.days2threads.length) that.expand_group_day(that.days2threads.length-1, true);
			that.live_update_info();
*/
			that.redisplay();

		},'json');
	};

	this.redisplay = function() {

		// Reset variables which are going to be recalculated when rebuilding
		// DOM in the below called functions.
		//
		that.total_messages = 0;
		that.runs = new Array();
		that.min_run = 0;
		that.max_run = 0;
		that.attachments = new Array();
		that.attachments_loader = new Array();
		var html = '';
		for(var day_idx = this.days2threads.length-1; day_idx >= 0; day_idx--)
			html += this.live_day2html(day_idx);
		$('#'+this.base+'-ms').html(html);

		if(this.days2threads.length) this.expand_group_day(this.days2threads.length-1, true);
		this.live_update_info();
	};
	this.live_update_info = function() {
		$('#'+this.base+'-ms-info').html(
			'<center><b>'+that.total_messages+'</b> messages'+
			(that.min_run ? ', runs: <b>'+that.min_run+'</b> .. <b>'+that.max_run+'</b>' : '')+
			'</center>'
		);
		$('#'+this.base+'-ms-updated').html('[ Last update on: <b>'+that.last_updated+'</b> ]');
	};
	this.live_day2html = function(day_idx) {
		var html = '';
		var day = that.days2threads[day_idx];
		var runs_messages_info = '<b>'+day.messages+'</b> messages'+( day.runs ? ', runs: <b>'+day.min_run+'</b> .. <b>'+day.max_run+'</b>' : '' );
		html +=
'<div class="el-l-m-d-hdr" id="'+this.base+'-m-d-hdr-'+day_idx+'" onclick="'+this.address+'.toggle_group_day('+day_idx+');">'+
'  <div style="float:left;">'+
'    <span class="toggler ui-icon ui-icon-triangle-1-e el-l-m-d-tgl" id="'+this.base+'-m-d-tgl-'+day_idx+'"></span>'+
'  </div>'+
'  <div style="float:left;" class="el-l-m-d">'+day.ymd+'</div>'+
'  <div style="float:right; margin-right:10px;" id="'+this.base+'-m-d-con-info-'+day_idx+'">'+runs_messages_info+'</div>'+
'  <div style="clear:both;"></div>'+
'</div>'+
'<div class="el-l-m-d-con el-l-m-d-hdn" id="'+this.base+'-m-d-con-'+day_idx+'" onmouseover="'+this.address+'.highlight_day('+"'"+day_idx+"'"+');">';

		for(var i = day.threads.length-1; i >= 0; i--) {
			var thread_idx = day.threads[i];
			html += that.live_thread2html(thread_idx);
   		}
			html +=
'</div>';
		return html;
	};

	this.create_live_message_dialogs = function(id, is_child) {
		var dlgs = $('#'+this.base+'-m-dlgs-'+id);
		if(dlgs.html() != '') return;
	    var select_tag_html = "<option> select tag </option>\n";
	    for( var i= 0; i < this.parent.used_tags.length; ++i )
	    	select_tag_html += '<option>'+this.parent.used_tags[i]+'</option>\n';

		var tags_html = '';
		var num_tags = 3;
	    for( var i = 0; i < num_tags; ++i )
	    	tags_html +=
'<div style="width: 100%;">'+
'  <select id="'+this.base+'-m-tags-library-'+i+'-'+id+'">'+select_tag_html+'</select>'+
'  <input type="text" class="elog-tag-name" id="'+this.base+'-m-tag-name-'+i+'-'+id+'" name="tag_name_'+i+'" value="" size=16 title="type new tag here or select a known one from the left" />'+
'  <input type="hidden" id="'+this.base+'-m-tag-value-'+i+'-'+id+'" name="tag_value_'+i+'" value="" />'+
'</div>';

		var entry = this.messages[id];
		var html =
'<div id="'+this.base+'-m-rdlg-'+id+'" class="el-l-m-rdlg el-l-m-dlg-hdn">'+
'  <div id="'+this.base+'-m-rdlg-'+id+'-info" style="color:maroon; position:relative; left:-10px; top:-15px;">Compose reply. Note the total limit of <b>25 MB</b> for attachments.</div>'+
'  <div style="float:left;">'+
'    <form id="elog-form-post-'+id+'" enctype="multipart/form-data" action="../logbook/NewFFEntry4portalJSON.php" method="post">'+
'      <input type="hidden" name="id" value="'+this.parent.exp_id+'" />'+
'      <input type="hidden" name="scope" value="message" />'+
'      <input type="hidden" name="message_id" value="'+id+'" />'+
'      <input type="hidden" name="run_id" value="" />'+
'      <input type="hidden" name="shift_id" value="" />'+
'      <input type="hidden" name="MAX_FILE_SIZE" value="25000000" />'+
'      <input type="hidden" name="num_tags" value="0" />'+
'      <input type="hidden" name="onsuccess" value="" />'+
'      <input type="hidden" name="relevance_time" value="" />'+
'      <textarea name="message_text" rows="12" cols="64" style="padding:4px;" title="the first line of the message body will be used as its subject" ></textarea>'+
'      <div style="margin-top: 10px;">'+
'        <div style="float:left;">'+ 
'          <div style="font-weight:bold;">Author:</div>'+
'          <input type="text" name="author_account" value="'+this.parent.author+'" size=32 style="padding:2px; margin-top:5px;" />'+
'        </div>'+
'        <div style="float:left; margin-left:20px;">'+ 
'          <div style="font-weight:bold;">Attachments:</div>'+
'          <div id="'+this.base+'-reply-as-'+id+'" style="margin-top:5px;">'+
'            <div>'+
'              <input type="file" name="file2attach_0" onchange="'+this.address+'.live_reply_add_attachment('+id+')" />'+
'              <input type="hidden" name="file2attach_0" value=""/ >'+
'            </div>'+
'          </div>'+
'        </div>'+
'        <div style="clear:both;"></div>'+
'      </div>'+
'    </form>'+
'  </div>'+
'  <div style="float:right; margin-left:5px;" class="s-b-con"><button id="'+this.base+'-m-re-c-'+id+'" onclick="'+this.address+'.live_message_reply_cancel('+id+');">cancel</button></div>'+
'  <div style="float:right; margin-left:5px;" class="s-b-con"><button id="'+this.base+'-m-re-s-'+id+'" onclick="'+this.address+'.live_message_reply_submit('+id+');">submit</button></div>'+
'  <div style="clear:both;"></div>'+
'</div>'+
'<div id="'+this.base+'-m-xtdlg-'+id+'" class="el-l-m-xtdlg el-l-m-dlg-hdn">'+
'  <div id="'+this.base+'-m-xtdlg-'+id+'-info" style="color:maroon; position:relative; left:-10px; top:-15px;">Select existing tags or define new tags to be added.</div>'+
'  <div style="float:left;">'+
'    <form id="elog-form-extend-tags-'+id+'" enctype="multipart/form-data" action="../logbook/ExtendFFEntry4portalJSON.php" method="post">'+
'      <input type="hidden" name="message_id" value="'+id+'" />'+
'      <input type="hidden" name="MAX_FILE_SIZE" value="25000000" />'+
'      <input type="hidden" name="num_tags" value="'+num_tags+'" />'+
'      <input type="hidden" name="onsuccess" value="" />'+
'      <div>'+
'        <div>'+tags_html+'</div>'+
'      </div>'+
'    </form>'+
'  </div>'+
'  <div style="float:right; margin-left:5px;" class="s-b-con"><button id="'+this.base+'-m-xt-c-'+id+'" onclick="'+this.address+'.live_message_extend_cancel('+id+', true);">cancel</button></div>'+
'  <div style="float:right; margin-left:5px;" class="s-b-con"><button id="'+this.base+'-m-xt-s-'+id+'" onclick="'+this.address+'.live_message_extend_submit('+id+', true);">submit</button></div>'+
'  <div style="clear:both;"></div>'+
'</div>'+
'<div id="'+this.base+'-m-xadlg-'+id+'" class="el-l-m-xadlg el-l-m-dlg-hdn">'+
'  <div id="'+this.base+'-m-xadlg-'+id+'-info" style="color:maroon; position:relative; left:-10px; top:-15px;">Select attachments to upload. Note the total limit of <b>25 MB</b>.</div>'+
'  <div style="float:left;">'+
'    <form id="elog-form-extend-attachments-'+id+'" enctype="multipart/form-data" action="../logbook/ExtendFFEntry4portalJSON.php" method="post">'+
'      <input type="hidden" name="message_id" value="'+id+'" />'+
'      <input type="hidden" name="MAX_FILE_SIZE" value="25000000" />'+
'      <input type="hidden" name="num_tags" value="0" />'+
'      <input type="hidden" name="onsuccess" value="" />'+
'      <div>'+
'        <div id="'+this.base+'-extend-as-'+id+'">'+
'          <div>'+
'            <input type="file" name="file2attach_0" onchange="'+this.address+'.live_extend_add_attachment('+id+')" />'+
'            <input type="hidden" name="file2attach_0" value=""/ >'+
'          </div>'+
'        </div>'+
'      </div>'+
'    </form>'+
'  </div>'+
'  <div style="float:right; margin-left:5px;" class="s-b-con"><button id="'+this.base+'-m-xa-c-'+id+'" onclick="'+this.address+'.live_message_extend_cancel('+id+', false);">cancel</button></div>'+
'  <div style="float:right; margin-left:5px;" class="s-b-con"><button id="'+this.base+'-m-xa-s-'+id+'" onclick="'+this.address+'.live_message_extend_submit('+id+', false);">submit</button></div>'+
'  <div style="clear:both;"></div>'+
'</div>';
		if(this.parent.editor) {
			html +=
'<div id="'+this.base+'-m-edlg-'+id+'" class="el-l-m-edlg el-l-m-dlg-hdn">'+
'  <div style="font-size:90%; text-decoration:underline; position:relative; left:-10px; top:-15px;">E d i t &nbsp; m e s s a g e</div>'+
'  <div style="float:left;">'+
'    <form id="elog-form-edit-'+id+'" enctype="multipart/form-data" action="../logbook/UpdateFFEntry4portal.php" method="post">'+
'      <input type="hidden" name="id" value="'+id+'" />'+
'      <input type="hidden" name="content_type" value="TEXT" />'+
'      <input type="hidden" name="onsuccess" value="" />'+
'      <textarea name="content" rows="12" cols="64" style="padding:4px;" title="the first line of the message body will be used as its subject" >'+entry.content+'</textarea>'+
'      <div style="font-weight:bold; margin-top: 10px;">Extra attachments:</div>'+
'      <div id="'+this.base+'-edit-as-'+id+'" style="margin-top:5px;">'+
'        <div>'+
'          <input type="file" name="file2attach_0" onchange="'+this.address+'.live_edit_add_attachment('+id+')" />'+
'          <input type="hidden" name="file2attach_0" value=""/ >'+
'        </div>'+
'      </div>'+
'    </form>'+
'  </div>'+
'  <div style="float:right; margin-left:5px;" class="s-b-con"><button class="el-l-m-ed-c" id="'+this.base+'-m-ed-c-'+id+'" onclick="'+this.address+'.live_message_edit_cancel('+id+');">cancel</button></div>'+
'  <div style="float:right; margin-left:5px;" class="s-b-con"><button class="el-l-m-ed-s" id="'+this.base+'-m-ed-s-'+id+'" onclick="'+this.address+'.live_message_edit_submit('+id+');">submit</button></div>'+
'  <div style="clear:both;"></div>'+
'</div>'+
'<div id="'+this.base+'-m-mdlg-'+id+'" class="el-l-m-mdlg el-l-m-dlg-hdn">'+
'  <div style="float:right; margin-left:5px;" class="s-b-con"><button class="el-l-m-mv-c" id="'+this.base+'-m-mv-c-'+id+'" onclick="'+this.address+'.live_message_move_cancel('+id+');">cancel</button></div>'+
'  <div style="float:right; margin-left:5px;" class="s-b-con"><button class="el-l-m-mv-s" id="'+this.base+'-m-mv-s-'+id+'" onclick="'+this.address+'.live_message_move_submit('+id+');">submit</button></div>'+
'  <div style="clear:both;"></div>'+
''+
'  Here be the dialog allowing to move the message into some other context and/or time.'+
'  Right now we create it automatically for each message after loading it.'+
'  We should probably optimize things by creating it at first use.'+
'  That way we will reduce the size of DOM'+
'</div>';
		}

		dlgs.html(html);

		$('#'+this.base+'-m-xt-s-'+id).button();
		$('#'+this.base+'-m-xt-c-'+id).button();
		$('#'+this.base+'-m-xa-s-'+id).button();
		$('#'+this.base+'-m-xa-c-'+id).button();

		$('#'+this.base+'-m-re-s-'+id).button();
		$('#'+this.base+'-m-re-c-'+id).button();

		if(this.parent.editor) {
			$('#'+this.base+'-m-ed-s-'+id).button();
			$('#'+this.base+'-m-ed-c-'+id).button();
			$('#'+this.base+'-m-mv-e-'+id).button();
			$('#'+this.base+'-m-mv-c-'+id).button();
		}
		$('#'+this.base+'-m-tags-library-0-'+id).change(function(ev) {
			var selectedIndex = $('#'+that.base+'-m-tags-library-0-'+id).attr('selectedIndex');
			if( selectedIndex == 0 ) return;
			$('#'+that.base+'-m-tag-name-0-'+id    ).val($('#'+that.base+'-m-tags-library-0-'+id).val());
			$('#'+that.base+'-m-tags-library-0-'+id).attr('selectedIndex', 0);
		});
		$('#'+this.base+'-m-tags-library-1-'+id).change(function(ev) {
			var selectedIndex = $('#'+that.base+'-m-tags-library-1-'+id).attr('selectedIndex');
			if( selectedIndex == 0 ) return;
			$('#'+that.base+'-m-tag-name-1-'+id    ).val($('#'+that.base+'-m-tags-library-1-'+id).val());
			$('#'+that.base+'-m-tags-library-1-'+id).attr('selectedIndex', 0);
		});
		$('#'+this.base+'-m-tags-library-2-'+id).change(function(ev) {
			var selectedIndex = $('#'+that.base+'-m-tags-library-2-'+id).attr('selectedIndex');
			if( selectedIndex == 0 ) return;
			$('#'+that.base+'-m-tag-name-2-'+id    ).val($('#'+that.base+'-m-tags-library-2-'+id).val());
			$('#'+that.base+'-m-tags-library-2-'+id).attr('selectedIndex', 0);
		});
	};

	this.live_enable_message_buttons = function(id, on) {
		var state = on ? 'enable' : 'disable';
		$('#'+this.base+'-m-xt-'+id).button(state);
		$('#'+this.base+'-m-xa-'+id).button(state);
		$('#'+this.base+'-m-re-'+id).button(state);
		$('#'+this.base+'-m-ed-'+id).button(state);
		$('#'+this.base+'-m-mv-'+id).button(state);
	};

	this.live_message_extend = function(id, is_child, add_tags_vs_attachments) {
	    that.create_live_message_dialogs(id, is_child);
	    if( add_tags_vs_attachments ) $('#'+this.base+'-m-xtdlg-'+id).removeClass('el-l-m-dlg-hdn').addClass('el-l-m-dlg-vis');
	    else                          $('#'+this.base+'-m-xadlg-'+id).removeClass('el-l-m-dlg-hdn').addClass('el-l-m-dlg-vis');
		that.live_enable_message_buttons(id, false);
	};
	this.live_message_reply = function(id, is_child) {
	    that.create_live_message_dialogs(id, is_child);
		$('#'+this.base+'-m-rdlg-'+id).removeClass('el-l-m-dlg-hdn').addClass('el-l-m-dlg-vis');
		that.live_enable_message_buttons(id, false);
		$('#'+this.base+'-m-rdlg-'+id+' textarea').focus();
	};
	this.live_message_edit = function(id, is_child) {
	    that.create_live_message_dialogs(id, is_child);
		$('#'+this.base+'-m-edlg-'+id).removeClass('el-l-m-dlg-hdn').addClass('el-l-m-dlg-vis');
		that.live_enable_message_buttons(id, false);
		$('#'+this.base+'-m-edlg-'+id+' textarea').focus();
	};
	this.live_message_move = function(id, is_child) {
	    that.create_live_message_dialogs(id, is_child);
		$('#'+this.base+'-m-mdlg-'+id).removeClass('el-l-m-dlg-hdn').addClass('el-l-m-dlg-vis');
		that.live_enable_message_buttons(id, false);
	};

	this.live_message_reply_cancel = function(id) {
		$('#'+this.base+'-m-rdlg-'+id).removeClass('el-l-m-dlg-vis').addClass('el-l-m-dlg-hdn');
		that.live_enable_message_buttons(id, true);
	};
	this.live_message_edit_cancel = function(id) {
		$('#'+this.base+'-m-edlg-'+id).removeClass('el-l-m-dlg-vis').addClass('el-l-m-dlg-hdn');
		that.live_enable_message_buttons(id, true);
	};
	this.live_message_move_cancel = function(id) {
		$('#'+this.base+'-m-mdlg-'+id).removeClass('el-l-m-dlg-vis').addClass('el-l-m-dlg-hdn');
		that.live_enable_message_buttons(id, true);
	};

	this.live_message_reply_submit = function(id) {

		var urlbase = window.location.href;
		var idx = urlbase.indexOf( '&' );
		if( idx > 0 ) urlbase = urlbase.substring( 0, idx );

		// Make sure there is anything to submit. Do not allow submitting
		// an empty message.
		//
		if( $('#elog-form-post-'+id+' textarea[name="message_text"]').val() == '' ) {
			this.parent.report_error('Can not post the empty message. Please put some text into the message box.');
			return;
		}

		var old_info = $('#'+this.base+'-m-rdlg-'+id+'-info').html();
		$('#'+this.base+'-m-rdlg-'+id+'-info').html('Posting. Please wait...');
		$('#'+this.base+'-m-re-c-'+id).button('disable');
		$('#'+this.base+'-m-re-s-'+id).button('disable');

		// Use JQuery AJAX Form plug-in to post the reply w/o reloading
		// the current page.
		//
		$('#elog-form-post-'+id).ajaxSubmit({
			success: function(data) {
				$('#elog-form-post-'+id+' textarea[name="message_text"]').val('');
				$('#elog-form-post-'+id+' input[name="author_account"]').val(that.parent.author);
				$('#'+that.base+'-reply-as-'+id).html(
'<div>'+
'  <input type="file" name="file2attach_0" onchange="'+that.address+'.live_reply_add_attachment('+id+')" />'+
'  <input type="hidden" name="file2attach_0" value=""/ >'+
'</div>'
				);
				if( data.Status != 'success' ) { that.parent.report_error( data.Message ); return; }
				var entry = data.Entry;

				// Find the parent message and add a new child to it.
				//
				// TODO: Should we find a way to serialize the JSON object into a string?
				//       A problem is that the children array updated below is also used
				//       from other locations of the code. And in those other locations
				//       elements of the children array are treated as strings, hence they're
				//       converted into valid JavaScript objects before passing them into
				//       the live_child2html() function
				//
				//       Another possibility would be to use typeof inside the code
				//       which uses the array of children to see if this is the string
				//       and retranslate it into JSON automatically, or do it immediatellty
				//       after loading.
				//
				that.messages[id].children.push( entry );

				// Then display the new child on the page and make it visible
				//
				var html = that.live_child2html(entry, that.messages[id].thread_idx);
				$('#'+that.base+'-m-c-'+id).prepend(html);

				$('.el-l-m-re').button();
				$('.el-l-m-ed').button();
				$('.el-l-m-mv').button();

				that.toggle_child(entry.id);

				that.live_message_reply_cancel(id);
			},
			complete: function() {
				$('#'+that.base+'-m-rdlg-'+id+'-info').html( old_info );
				$('#'+that.base+'-m-re-c-'+id).button('enable');
				$('#'+that.base+'-m-re-s-'+id).button('enable');
			},
			dataType: 'json'
		});

	};
	this.live_message_edit_submit = function(id) {
		var urlbase = window.location.href;
		var idx = urlbase.indexOf( '&' );
		if( idx > 0 ) urlbase = urlbase.substring( 0, idx );
		$('#elog-form-edit-'+id+' input[name="onsuccess"]').val(urlbase+'&app=elog');

		$('#elog-form-edit-'+id).trigger( 'submit' );
	};
	this.live_message_move_submit = function(id) {
		$('#'+this.base+'-m-mdlg-'+id).removeClass('el-l-m-dlg-vis').addClass('el-l-m-dlg-hdn');
		that.live_enable_message_buttons(id, true);
	};

	this.live_reply_add_attachment = function(id) {
		var num = $('#'+this.base+'-reply-as-'+id+' > div').size();
		$('#'+this.base+'-reply-as-'+id).append(
'  <div>'+
'    <input type="file" name="file2attach_'+num+'" onchange="'+this.address+'.live_reply_add_attachment('+id+')" />'+
'    <input type="hidden" name="file2attach_'+num+'" value=""/ >'+
'  </div>'
		);
	};
	this.live_edit_add_attachment = function(id) {
		var num = $('#'+this.base+'-edit-as-'+id+' > div').size();
		$('#'+this.base+'-edit-as-'+id).append(
'  <div>'+
'    <input type="file" name="file2attach_'+num+'" onchange="'+this.address+'.live_edit_add_attachment('+id+')" />'+
'    <input type="hidden" name="file2attach_'+num+'" value=""/ >'+
'  </div>'
		);
	};


	this.live_run_reply_cancel = function(id) {
		$('#'+this.base+'-r-rdlg-'+id).removeClass('el-l-r-dlg-vis').addClass('el-l-r-dlg-hdn');
		that.live_enable_run_buttons(id, true);
	};
	this.live_run_reply_submit = function(id) {

		var urlbase = window.location.href;
		var idx = urlbase.indexOf( '&' );
		if( idx > 0 ) urlbase = urlbase.substring( 0, idx );

		// Make sure there is anything to submit. Do not allow submitting
		// an empty message.
		//
		if( $('#elog-form-post-'+id+' textarea[name="message_text"]').val() == '' ) {
			this.parent.report_error('Can not post the empty message. Please put some text into the message box.');
			return;
		}

		// Use JQuery AJAX Form plug-in to post the reply w/o reloading
		// the current page.
		//
		var old_info = $('#'+this.base+'-m-rdlg-'+id+'-info').html();
		$('#'+this.base+'-r-rdlg-'+id+'-info').html('Posting. Please wait...');
		$('#'+this.base+'-r-re-c-'+id).button('disable');
		$('#'+this.base+'-r-re-s-'+id).button('disable');

		$('#elog-form-post-'+id).ajaxSubmit({
			success: function(data) {
				$('#elog-form-post-'+id+' textarea[name="message_text"]').val('');
				$('#elog-form-post-'+id+' input[name="author_account"]').val(that.parent.author);
				$('#'+that.base+'-reply-as-'+id).html(
'<div>'+
'  <input type="file" name="file2attach_0" onchange="'+that.address+'.live_reply_add_attachment('+id+')" />'+
'  <input type="hidden" name="file2attach_0" value=""/ >'+
'</div>'
				);
				if( data.Status != 'success' ) { that.parent.report_error( data.Message ); return; }
				that.live_run_reply_cancel(id);
			},
			complete: function() {
				$('#'+that.base+'-r-rdlg-'+id+'-info').html( old_info );
				$('#'+that.base+'-r-re-c-'+id).button('enable');
				$('#'+that.base+'-r-re-s-'+id).button('enable');
			},
			dataType: 'json'
		});

	};

	this.live_extend_add_attachment = function(id) {
		var num = $('#'+this.base+'-extend-as-'+id+' > div').size();
		$('#'+this.base+'-extend-as-'+id).append(
'  <div>'+
'    <input type="file" name="file2attach_'+num+'" onchange="'+this.address+'.live_extend_add_attachment('+id+')" />'+
'    <input type="hidden" name="file2attach_'+num+'" value=""/ >'+
'  </div>'
		);
	};
	this.live_message_extend_cancel = function(id, add_tags_vs_attachments ) {
		if( add_tags_vs_attachments ) $('#'+this.base+'-m-xtdlg-'+id).removeClass('el-l-m-dlg-vis').addClass('el-l-m-dlg-hdn');
		else                          $('#'+this.base+'-m-xadlg-'+id).removeClass('el-l-m-dlg-vis').addClass('el-l-m-dlg-hdn');
		that.live_enable_message_buttons(id, true);
	};
	this.live_message_extend_submit = function(id, add_tags_vs_attachments) {

		var urlbase = window.location.href;
		var idx = urlbase.indexOf( '&' );
		if( idx > 0 ) urlbase = urlbase.substring( 0, idx );

		// Use JQuery AJAX Form plug-in to post the reply w/o reloading
		// the current page.
		//
		if( add_tags_vs_attachments ) {

			var old_info = $('#'+this.base+'-m-xtdlg-'+id+'-info').html();
			$('#'+this.base+'-m-xtdlg-'+id+'-info').html('Submitting. Please wait...');
			$('#'+this.base+'-m-xt-c-'+id).button('disable');
			$('#'+this.base+'-m-xt-s-'+id).button('disable');

			$('#elog-form-extend-tags-'+id).ajaxSubmit({
				success: function(data) {
					if( data.Status != 'success' ) { that.parent.report_error( data.Message ); return; }
					var tags = data.Extended.tags;
					for( var i = 0; i < tags.length; i++ )
						that.messages[id].tags.unshift(tags[i]);

					that.expand_message(that.messages[id].thread_idx, true);
				},
				complete: function() {
					$('#'+that.base+'-m-xtdlg-'+id+'-info').html( old_info );
					$('#'+that.base+'-m-xt-c-'+id).button('enable');
					$('#'+that.base+'-m-xt-s-'+id).button('enable');
				},
				dataType: 'json'
			});
		} else {

			var old_info = $('#'+this.base+'-m-xadlg-'+id+'-info').html();
			$('#'+this.base+'-m-xadlg-'+id+'-info').html('Uploading. Please wait...');
			$('#'+this.base+'-m-xa-c-'+id).button('disable');
			$('#'+this.base+'-m-xa-s-'+id).button('disable');

			$('#elog-form-extend-attachments-'+id).ajaxSubmit({
				success: function(data) {
					$('#'+that.base+'-extend-as-'+id).html(
'<div>'+
'  <input type="file" name="file2attach_0" onchange="'+that.address+'.live_extend_add_attachment('+id+')" />'+
'  <input type="hidden" name="file2attach_0" value=""/ >'+
'</div>'
					);
					if( data.Status != 'success' ) { that.parent.report_error( data.Message ); return; }
					var attachments = data.Extended.attachments;
					for( var i = 0; i < attachments.length; i++ )
						that.messages[id].attachments.push(attachments[i]);

					that.expand_message(that.messages[id].thread_idx, true);
				},
				complete: function() {
					$('#'+that.base+'-m-xadlg-'+id+'-info').html( old_info );
					$('#'+that.base+'-m-xa-c-'+id).button('enable');
					$('#'+that.base+'-m-xa-s-'+id).button('enable');
				},
				dataType: 'json'
			});
		}
	};




	this.live_edit_add_attachment = function(id) {
		var num = $('#'+this.base+'-edit-as-'+id+' > div').size();
		$('#'+this.base+'-edit-as-'+id).append(
'  <div>'+
'    <input type="file" name="file2attach_'+num+'" onchange="'+this.address+'.live_edit_add_attachment('+id+')" />'+
'    <input type="hidden" name="file2attach_'+num+'" value=""/ >'+
'  </div>'
		);
	};

	this.create_live_run_dialogs = function(idx) {
		var entry = this.threads[idx];
		var id    = entry.id;
		var dlgs  = $('#'+this.base+'-r-dlgs-'+id);
		if(dlgs.html() != '') return;
		var html =
'<div id="'+this.base+'-r-rdlg-'+id+'" class="el-l-r-rdlg el-l-r-dlg-hdn">'+
'  <div id="'+this.base+'-r-rdlg-'+id+'-info" style="color:maroon; position:relative; left:-10px; top:-15px;">Compose message. Note the total limit of <b>25 MB</b> for attachments.</div>'+
'  <div style="float:left;">'+
'    <form id="elog-form-post-'+id+'" enctype="multipart/form-data" action="../logbook/NewFFEntry4portalJSON.php" method="post">'+
'      <input type="hidden" name="id" value="'+this.parent.exp_id+'" />'+
'      <input type="hidden" name="scope" value="run" />'+
'      <input type="hidden" name="message_id" value="" />'+
'      <input type="hidden" name="run_id" value="'+entry.run_id+'" />'+
'      <input type="hidden" name="shift_id" value="" />'+
'      <input type="hidden" name="MAX_FILE_SIZE" value="25000000" />'+
'      <input type="hidden" name="num_tags" value="0" />'+
'      <input type="hidden" name="onsuccess" value="" />'+
'      <input type="hidden" name="relevance_time" value="" />'+
'      <textarea name="message_text" rows="12" cols="64" style="padding:4px;" title="the first line of the message body will be used as its subject" ></textarea>'+
'      <div style="margin-top: 10px;">'+
'        <div style="float:left;">'+ 
'          <div style="font-weight:bold;">Author:</div>'+
'          <input type="text" name="author_account" value="'+this.parent.author+'" size=32 style="padding:2px; margin-top:5px;" />'+
'        </div>'+
'        <div style="float:left; margin-left:20px;">'+ 
'          <div style="font-weight:bold;">Attachments:</div>'+
'          <div id="'+this.base+'-reply-as-'+id+'" style="margin-top:5px;">'+
'            <div>'+
'              <input type="file" name="file2attach_0" onchange="'+this.address+'.live_reply_add_attachment('+id+')" />'+
'              <input type="hidden" name="file2attach_0" value=""/ >'+
'            </div>'+
'          </div>'+
'        </div>'+
'        <div style="clear:both;"></div>'+
'      </div>'+
'    </form>'+
'  </div>'+
'  <div style="float:right; margin-left:5px;" class="s-b-con"><button id="'+this.base+'-r-re-c-'+id+'" onclick="'+this.address+'.live_run_reply_cancel('+id+');">cancel</button></div>'+
'  <div style="float:right; margin-left:5px;" class="s-b-con"><button id="'+this.base+'-r-re-s-'+id+'" onclick="'+this.address+'.live_run_reply_submit('+id+');">submit</button></div>'+
'  <div style="clear:both;"></div>'+
'</div>';

		dlgs.html(html);

		$('#'+this.base+'-r-re-s-'+id).button();
		$('#'+this.base+'-r-re-c-'+id).button();

	};
	this.live_enable_run_buttons = function(id, on) {
		var state = on ? 'enable' : 'disable';
		$('#'+this.base+'-r-re-'+id).button(state);
	};
	this.live_run_reply = function(idx) {
	    that.create_live_run_dialogs(idx);
		var entry = this.threads[idx];
		var id    = entry.id;
		$('#'+this.base+'-r-rdlg-'+id).removeClass('el-l-r-dlg-hdn').addClass('el-l-r-dlg-vis');
		that.live_enable_run_buttons(id, false);
		$('#'+this.base+'-r-rdlg-'+id+' textarea').focus();
	};

	this.live_thread2html = function(thread_idx) {
		var html = '';
		var entry = that.threads[thread_idx];
		if(!entry.is_run) {
			that.messages[entry.id] = entry;
			that.total_messages++;
			html +=
'  <div class="el-l-m-hdr" id="'+this.base+'-m-hdr-'+thread_idx+'" onclick="'+this.address+'.toggle_message('+thread_idx+');">'+
'    <div style="float:left;"><span class="toggler ui-icon ui-icon-triangle-1-e el-l-m-tgl" id="'+this.base+'-m-tgl-'+entry.id+'"></span></div>'+
'    <div style="float:left;" class="el-l-m-time">'+entry.hms+'</div>'+
'    <div style="float:left;" class="el-l-m-author">'+entry.author+'</div>'+
'    <div style="float:left; margin-left:10px;" class="el-l-m-subj">'+entry.subject+'</div>'+id_sign(entry)+run_sign(entry.run_num)+message_signs(entry)+
'    <div style="clear:both;"></div>'+
'  </div>'+
'  <div class="el-l-m-con el-l-m-hdn" id="'+this.base+'-m-con-'+entry.id+'"></div>';

			// The message contents container will be dynamically initialized at its
			// first use (call to the toggle_message() function.
			//
			;

		} else {
        	that.runs.push(thread_idx);
			if((that.min_run == 0) || (entry.run_num < that.min_run)) that.min_run = entry.run_num;
			if((that.max_run == 0) || (entry.run_num > that.max_run)) that.max_run = entry.run_num;
			html +=
'  <div class="el-l-r-hdr" id="'+this.base+'-m-hdr-'+thread_idx+'" onclick="'+this.address+'.toggle_run('+thread_idx+');">'+
'    <div style="float:left;"><span class="toggler ui-icon ui-icon-triangle-1-e el-l-r-tgl" id="'+this.base+'-r-tgl-'+entry.id+'"></span></div>'+
'    <div style="float:left;" class="el-l-m-time">'+entry.hms+'</div>'+
'    <div style="float:left;" class="el-l-m-author">'+entry.author+'</div>'+
'    <div style="float:left; margin-left:10px;" class="el-l-m-subj">'+entry.subject+'</div>'+
'    <div style="clear:both;"></div>'+
'  </div>'+
'  <div class="el-l-r-con el-l-r-hdn" id="'+this.base+'-r-con-'+entry.id+'"></div>';

			// The run contents container will be dynamically initialized at its
			// first use (call to the toggle_run() function.
			//
			;
		}
		return html;
	};

	this.live_child2html = function(entry, thread_idx) {
		that.messages[entry.id] = entry;
		var html =
'<div class="el-l-c-hdr" onclick="'+this.address+'.toggle_child('+entry.id+');">'+
'  <div style="float:left;"><span class="toggler ui-icon ui-icon-triangle-1-e el-l-c-tgl" id="'+this.base+'-c-tgl-'+entry.id+'"></span></div>'+
'  <div style="float:left;" class="el-l-c-time">'+entry.relevance_time+'</div>'+
'  <div style="float:left;" class="el-l-c-author">'+entry.author+'</div>'+
'  <div style="float:left; margin-left:10px;" class="el-l-c-subj">'+entry.subject+'</div>'+id_sign(entry)+message_signs(entry)+
'  <div style="clear:both;"></div>'+
'</div>'+
'<div class="el-l-c-con el-l-c-hdn" id="'+this.base+'-c-con-'+entry.id+'">'+
'  <div class="el-l-c-body">'+
'    <div style="float:left; font-size:12px; width:100%; overflow:auto;">'+entry.html1+'</div>';
		if(this.parent.editor)
			html +=
'    <div style="float:right;" class="s-b-con"><button class="el-l-m-ed"  id="'+this.base+'-m-ed-'+entry.id+'"  onclick="'+this.address+'.live_message_edit('+entry.id+',true);">edit</button></div>';
		html +=
'    <div style="float:right;" class="s-b-con"><button class="el-l-m-re" id="'+this.base+'-m-re-'+entry.id+'" onclick="'+this.address+'.live_message_reply('+entry.id+',true);">reply</button></div>'+
'    <div style="clear:both;"></div>'+
'      <div id="'+this.base+'-m-dlgs-'+entry.id+'"></div>'+
'  </div>';
		that.attachments[entry.id] = entry.attachments;

		var attachments_html = '';
		for( var i in entry.attachments) {
			var a = entry.attachments[i];
			that.attachments_loader[a.id] = {loaded: false, descr: a};
			attachments_html +=
'    <div style="float:left;" class="el-l-a">'+
'      <div style="float:left;">'+
'        <span class="toggler ui-icon ui-icon-triangle-1-e el-l-a-tgl" id="'+this.base+'-a-tgl-'+a.id+'" onclick="'+this.address+'.toggle_attachment('+a.id+');"></span>'+
'      </div>'+
'      <div style="float:left;" class="el-l-a-dsc"><a class="link" href="../logbook/attachments/'+a.id+'/preview"  target="_blank">'+a.description+'</a></div>'+
'      <div style="float:left; margin-left:10px;" class="el-l-a-info">( type: <b>'+a.type+'</b> size: <b>'+a.size+'</b> )</div>'+
'      <div style="clear:both;"></div>'+
'      <div class="el-l-a-con el-l-a-hdn" id="'+this.base+'-a-con-'+a.id+'">'+
'      </div>'+
'    </div>';
		}
		if(attachments_html) html +=
'  <div class="el-l-m-as">'+attachments_html+
'    <div style="clear:both;"></div>'+
'  </div>';
		html +=
'  <div id="'+this.base+'-m-c-'+entry.id+'">';

		var children = entry.children;
		for(var i in children) html += that.live_child2html(eval("("+children[i]+")"), thread_idx);

		html +=
'  </div>'+
'</div>';
		return html;
	};

	function message_signs(entry) {
		var html = '<div style="float:right; margin-right:10px;">';
		for(var i=0; i < entry.attachments.length; i++) html += '<img src="../logbook/images/attachment.png" height="18" />';
		for(var i=0; i < entry.tags.length; i++) html += '&nbsp;<sup><b>T</b></sup>';
		if(entry.children.length) html +=  '&nbsp;<sup><b>&crarr;</b></sup>';
		html += '</div>';
		return html;
	}
	function id_sign(entry) {
		var html = '<div style="float:right; margin-left:10px; margin-top:2px; padding-left:2px; padding-right:2px; background-color:#ffffff;">'+entry.id+'</div>';
		return html;
	}

	function run_sign(num) {
		if( num ) {
			var html = '<div style="float:right; margin-right:10px;"><sup><b>run: '+num+'</b></sup></div>';
			return html;
		}
		return '';
	}

	/* ATTENTION: Watch for dependencies! These functiona will call a function from
	 *            the application context.
	 *
	 * TODO: Re-evaluate this code to see if the dependencies are properly
	 *       designed and used.
	 */
	this.highlight_day = function(idx) {
		return;
		if(that.current_day == idx) return;
		that.current_day = idx;
		var day = that.days2threads[idx];
		applications['p-appl-elog'].context3 = '<a class="link" href="#'+this.base+'-m-d-hdr-'+idx+'" title="go to the day header">'+that.days2threads[idx].ymd+'</a>';
		set_context(applications['p-appl-elog']);
	};

	this.dim_day = function() {
		return;
		if(that.current_day == null) return;
		that.current_day = null;
		applications['p-appl-elog'].context3 ='';
		set_context(applications['p-appl-elog']);
	};
}

/**
 * A simplified object for displaying e-log messages associated with the specified run.
 * 
 * @object_address - the full path name to an instance of the object (used for global references from collback functions
 * @param parent - parent object
 * @param element_base - the base of an element where to grow the DOM
 * @run_num - the run number
 * @return
 */
function elog_message_viewer4run_create(object_address, parent_object, element_base, run_num) {

	this.address = object_address;
	this.parent = parent_object;
	this.base = element_base;
	this.run_num = run_num;

	this.threads = null;
	this.messages = null;
	this.runs = null;
	this.attachments = null;
	this.attachments_loader = null;

	var that = this;
/*
	this.expand_message = function(idx, on) {
		var entry = that.threads[idx];
		var toggler='#'+that.base+'-m-tgl-'+entry.id;
		var container='#'+that.base+'-m-con-'+entry.id;

		// Initialize the thread container if this is the first call
		// to the function for for the message.
		//
		if( $(container).html() == '' ) {
			var html =
'    <div class="el-l-m-body">';
			if(this.parent.editor)
				html +=
'      <div style="float:right;" class="s-b-con"><button class="el-l-m-mv"  id="'+this.base+'-m-mv-'+entry.id+'"  onclick="'+this.address+'.live_message_move('+entry.id+',false);">move</button></div>'+
'      <div style="float:right;" class="s-b-con"><button class="el-l-m-ed"  id="'+this.base+'-m-ed-'+entry.id+'"  onclick="'+this.address+'.live_message_edit('+entry.id+',false);">edit</button></div>';
			html +=
'      <div style="float:right;" class="s-b-con"><button class="el-l-m-re" id="'+this.base+'-m-re-'+entry.id+'" onclick="'+this.address+'.live_message_reply('+entry.id+',false);">reply</button></div>'+
'      <div style="float:left; font-size:12px; width:100%; overflow:auto;">'+entry.html1+'</div>'+
'      <div style="clear:both;"></div>'+
'      <div id="'+this.base+'-m-dlgs-'+entry.id+'"></div>'+
'    </div>';
			that.attachments[entry.id] = entry.attachments;
			var attachments_html = '';
			for(var k in entry.attachments) {
				var a = entry.attachments[k];
				that.attachments_loader[a.id] = {loaded: false, descr: a};
				attachments_html +=
'      <div style="float:left;" class="el-l-a">'+
'        <div style="float:left;">'+
'          <span class="toggler ui-icon ui-icon-triangle-1-e el-l-a-tgl" id="'+this.base+'-a-tgl-'+a.id+'" onclick="'+this.address+'.toggle_attachment('+a.id+');"></span>'+
'        </div>'+
'        <div style="float:left;" class="el-l-a-dsc"><a class="link" href="../logbook/attachments/'+a.id+'/'+a.description+'"  target="_blank">'+a.description+'</a></div>'+
'        <div style="float:left; margin-left:10px;" class="el-l-a-info">( type: <b>'+a.type+'</b> size: <b>'+a.size+'</b> )</div>'+
'        <div style="clear:both;"></div>'+
'        <div class="el-l-a-con el-l-a-hdn" id="'+this.base+'-a-con-'+a.id+'">'+
'        </div>'+
'      </div>';
			}
			if(attachments_html) html +=
'    <div class="el-l-m-as">'+attachments_html+
'      <div style="clear:both;"></div>'+
'    </div>';
			var tags_html = '';
			for(var k in entry.tags) {
				if(tags_html) tags_html += ', ';
				tags_html += entry.tags[k].tag;
			}
			if(tags_html) html +=
'    <div class="el-l-m-tags">'+
'      <b><i>keywords: </i></b>'+tags_html+
'    </div>';
			html +=
'    <div id="'+this.base+'-m-c-'+entry.id+'">';
			for(var k in entry.children) html += that.live_child2html(eval("("+entry.children[k]+")"));
			html +=
'    </div>'+
'  </div>';
			$(container).html(html);

			$(container).find('.el-l-m-re').button();
			$(container).find('.el-l-m-ed').button();
			$(container).find('.el-l-m-mv').button();
		}
		if(on) {
			$(toggler).removeClass('ui-icon-triangle-1-e').addClass('ui-icon-triangle-1-s');
			$(container).removeClass('el-l-m-hdn').addClass('el-l-m-vis');
			for(var i = 0; i < entry.attachments.length; i++) {
				this.expand_attachment(entry.attachments[i].id, true);
			}
			for(var i = 0; i < entry.children.length; i++) {
				var child = eval( "("+entry.children[i]+")" );
				this.expand_child(child.id, true);
			}
		} else {
			$(toggler).removeClass('ui-icon-triangle-1-s').addClass('ui-icon-triangle-1-e');
			$(container).removeClass('el-l-m-vis').addClass('el-l-m-hdn');
		}
	};
*/
	this.expand_message = function(idx, on) {
		var entry = that.threads[idx];
		var toggler='#'+that.base+'-m-tgl-'+entry.id;
		var container='#'+that.base+'-m-con-'+entry.id;

		// REIMPLEMENTED THIS:
		//   Initialize the thread container if this is the first call
		//   to the function for for the message.
		//
		//   if( $(container).html() == '' ) {
		//
		// TO THIS:
		//    Always recreate the thread when opening it to make sure the new contents
		//    gets properly displayed.
		//
		// TODO: Do the same for the message viewer in runs.
		//
		if(on) {
			entry.thread_idx = idx;
			var html =
'    <div class="el-l-m-body">';
			if(this.parent.editor)
				html +=
'      <div style="float:right;" class="s-b-con"><button class="el-l-m-mv"  id="'+this.base+'-m-mv-'+entry.id+'"  onclick="'+this.address+'.live_message_move('+entry.id+',false);">move</button></div>'+
'      <div style="float:right;" class="s-b-con"><button class="el-l-m-ed"  id="'+this.base+'-m-ed-'+entry.id+'"  onclick="'+this.address+'.live_message_edit('+entry.id+',false);">edit</button></div>';
			html +=
'      <div style="float:right; margin-left:5px;" class="s-b-con"><button class="el-l-m-xt" id="'+this.base+'-m-xt-'+entry.id+'" onclick="'+this.address+'.live_message_extend('+entry.id+',false, true);" title="add more tags to the message">+ tags</button></div>'+
'      <div style="float:right; margin-left:5px;" class="s-b-con"><button class="el-l-m-xa" id="'+this.base+'-m-xa-'+entry.id+'" onclick="'+this.address+'.live_message_extend('+entry.id+',false, false);" title="add more attachments to the message">+ attachments</button></div>'+
'      <div style="float:right;"                  class="s-b-con"><button class="el-l-m-re" id="'+this.base+'-m-re-'+entry.id+'" onclick="'+this.address+'.live_message_reply('+entry.id+',false);" title="reply to the message">reply</button></div>'+
'      <div style="float:left; font-size:12px; width:100%; overflow:auto;">'+entry.html1+'</div>'+
'      <div style="clear:both;"></div>'+
'      <div id="'+this.base+'-m-dlgs-'+entry.id+'"></div>'+
'    </div>';
			that.attachments[entry.id] = entry.attachments;
			var attachments_html = '';
			for(var k in entry.attachments) {
				var a = entry.attachments[k];
				that.attachments_loader[a.id] = {loaded: false, descr: a};
				attachments_html +=
'      <div style="float:left;" class="el-l-a">'+
'        <div style="float:left;">'+
'          <span class="toggler ui-icon ui-icon-triangle-1-e el-l-a-tgl" id="'+this.base+'-a-tgl-'+a.id+'" onclick="'+this.address+'.toggle_attachment('+a.id+');"></span>'+
'        </div>'+
'        <div style="float:left;" class="el-l-a-dsc"><a class="link" href="../logbook/attachments/'+a.id+'/'+a.description+'"  target="_blank">'+a.description+'</a></div>'+
'        <div style="float:left; margin-left:10px;" class="el-l-a-info">( type: <b>'+a.type+'</b> size: <b>'+a.size+'</b> )</div>'+
'        <div style="clear:both;"></div>'+
'        <div class="el-l-a-con el-l-a-hdn" id="'+this.base+'-a-con-'+a.id+'">'+
'        </div>'+
'      </div>';
			}
			if(attachments_html) html +=
'    <div class="el-l-m-as">'+attachments_html+
'      <div style="clear:both;"></div>'+
'    </div>';
			var tags_html = '';
			for(var k in entry.tags) {
				if(tags_html) tags_html += ', ';
				tags_html += entry.tags[k].tag;
			}
			if(tags_html) html +=
'    <div class="el-l-m-tags">'+
'      <b><i>keywords: </i></b>'+tags_html+
'    </div>';
			html +=
'    <div id="'+this.base+'-m-c-'+entry.id+'">';
			for(var k in entry.children) {
				var child = entry.children[k];
				if( typeof child == 'string' ) html += that.live_child2html(eval("("+child+")"), idx);
				else                           html += that.live_child2html(child, idx);
			}
			html +=
'    </div>'+
'  </div>';
			$(container).html(html);

			$(container).find('.el-l-m-xt').button();
			$(container).find('.el-l-m-xa').button();
			$(container).find('.el-l-m-re').button();
			$(container).find('.el-l-m-ed').button();
			$(container).find('.el-l-m-mv').button();

			$(toggler).removeClass('ui-icon-triangle-1-e').addClass('ui-icon-triangle-1-s');
			$(container).removeClass('el-l-m-hdn').addClass('el-l-m-vis');
			for(var i = 0; i < entry.attachments.length; i++) {
				this.expand_attachment(entry.attachments[i].id, true);
			}
			for(var i = 0; i < entry.children.length; i++) {
				var child = entry.children[i];
				var child_entry = ( typeof child == 'string' ) ? eval( "("+child+")" ) : child;
				this.expand_child(child_entry.id, true);
			}

		} else {
			$(container).html('');
			$(toggler).removeClass('ui-icon-triangle-1-s').addClass('ui-icon-triangle-1-e');
			$(container).removeClass('el-l-m-vis').addClass('el-l-m-hdn');
		}
	};

	this.expand_all_messages = function() {
		for(var i = that.threads.length-1; i >= 0; i--)
			this.expand_message(i, true);
	};

	this.toggle_message = function(idx) {
		var entry = that.threads[idx];
		var container='#'+that.base+'-m-con-'+entry.id;
		this.expand_message(idx, $(container).hasClass('el-l-m-hdn'));
	};

	this.expand_child = function(id, on) {
		var entry = that.messages[id];
		var toggler='#'+this.base+'-c-tgl-'+id;
		var container='#'+this.base+'-c-con-'+id;
		if(on) {
			$(toggler).removeClass('ui-icon-triangle-1-e').addClass('ui-icon-triangle-1-s');
			$(container).removeClass('el-l-c-hdn').addClass('el-l-c-vis');
			for(var i = 0; i < entry.attachments.length; i++) {
				this.expand_attachment(entry.attachments[i].id, true);
			}
			for(var i = 0; i < entry.children.length; i++) {
				var child = eval( "("+entry.children[i]+")" );
				this.expand_child(child.id, true);
			}
		} else {
			$(toggler).removeClass('ui-icon-triangle-1-s').addClass('ui-icon-triangle-1-e');
			$(container).removeClass('el-l-c-vis').addClass('el-l-c-hdn');
		}
	};

	this.toggle_child = function(id) {
		var container='#'+this.base+'-c-con-'+id;
		this.expand_child(id, $(container).hasClass('el-l-c-hdn'));
	};

	this.expand_attachment = function(id, on) {
		var toggler='#'+this.base+'-a-tgl-'+id;
		var container='#'+this.base+'-a-con-'+id;
		if(on) {
			var a = that.attachments_loader[id];
			if(!a.loaded) {
				a.loaded = true;
			    var html = '<a href="../logbook/attachments/'+id+'/'+a.descr.description+'" target="_blank"><img src="../logbook/attachments/preview/'+id+'" /></a>';
				$(container).html(html);
			}
			$(toggler).removeClass('ui-icon-triangle-1-e').addClass('ui-icon-triangle-1-s');
			$(container).removeClass('el-l-a-hdn').addClass('el-l-a-vis');
		} else {
			$(toggler).removeClass('ui-icon-triangle-1-s').addClass('ui-icon-triangle-1-e');
			$(container).removeClass('el-l-a-vis').addClass('el-l-a-hdn');
		}
	};

	this.toggle_attachment = function(id) {
		var container='#'+this.base+'-a-con-'+id;
		this.expand_attachment(id, $(container).hasClass('el-l-a-hdn'));
	};

	this.toggle_run = function(idx) {
		var entry=this.threads[idx];
		var toggler='#'+this.base+'-r-tgl-'+entry.id;
		var container='#'+this.base+'-r-con-'+entry.id;
		if( $(container).hasClass('el-l-r-hdn')) {
			$(toggler).removeClass('ui-icon-triangle-1-e').addClass('ui-icon-triangle-1-s');
			$(container).removeClass('el-l-r-hdn').addClass('el-l-r-vis');
			if(entry.loaded) return;
			$('#'+this.base+'-r-con-'+entry.id).html('Loading...');
			$.get('../logbook/DisplayRunParams.php',{id:entry.run_id},function(data) {
				var html =
'<div style="float:right;" class="s-b-con"><button class="el-l-r-re" id="'+that.base+'-r-re-'+entry.id+'" onclick="'+that.address+'.live_run_reply('+idx+');">reply</button></div>'+
'<div style="clear:both;"></div>'+
'<div id="'+that.base+'-r-dlgs-'+entry.id+'"></div>'+
'<div style="width:800px; height:300px; overflow:auto; background-color:#ffffff; ">'+data+'</div>';
				$('#'+that.base+'-r-con-'+entry.id).html(html);
				$('#'+that.base+'-r-re-'+entry.id).button();
				entry.loaded = true;
			});
		} else {
			$(toggler).removeClass('ui-icon-triangle-1-s').addClass('ui-icon-triangle-1-e');
			$(container).removeClass('el-l-r-vis').addClass('el-l-r-hdn');
		}
	};
	this.collapse_run = function(idx) {
	   	var entry=this.threads[idx];
		var toggler='#'+this.base+'-r-tgl-'+entry.id;
		var container='#'+this.base+'-r-con-'+entry.id;
		if( $(container).hasClass('el-l-r-vis')) {
			$(toggler).removeClass('ui-icon-triangle-1-s').addClass('ui-icon-triangle-1-e');
			$(container).removeClass('el-l-r-vis').addClass('el-l-r-hdn');
		}
	};
	this.reload = function() {

		var params = {
			id: this.parent.exp_id,
			scope: 'experiment',
			search_in_messages: 1,
			search_in_tags: 1,
			search_in_values: 1,
			posted_at_experiment: 1,
			posted_at_shifts: 1,
			posted_at_runs: 1,
			range_of_runs: this.run_num,
			inject_runs: '',
			format: 'detailed'
		};
		$.get('../logbook/Search.php',params,function(data) {

			if(data.ResultSet.Status != 'success') { that.parent.report_error( data.ResultSet.Message ); return; }

			that.threads = data.ResultSet.Result;
			that.messages = new Array();
			that.runs = new Array();
			that.attachments = new Array();
			that.attachments_loader = new Array();

			var html = '';
			for(var i=that.threads.length-1; i >= 0 ; --i)
				html += that.live_thread2html(i);
			$('#'+that.base).html(html);

		},'json');
	};
	this.create_live_message_dialogs = function(id, is_child) {
		var dlgs = $('#'+this.base+'-m-dlgs-'+id);
		if(dlgs.html() != '') return;
	    var select_tag_html = "<option> select tag </option>\n";
	    for( var i= 0; i < this.parent.used_tags.length; ++i )
	    	select_tag_html += '<option>'+this.parent.used_tags[i]+'</option>\n';

		var tags_html = '';
		var num_tags = 3;
	    for( var i = 0; i < num_tags; ++i )
	    	tags_html +=
'<div style="width: 100%;">'+
'  <select id="'+this.base+'-m-tags-library-'+i+'-'+id+'">'+select_tag_html+'</select>'+
'  <input type="text" class="elog-tag-name" id="'+this.base+'-m-tag-name-'+i+'-'+id+'" name="tag_name_'+i+'" value="" size=16 title="type new tag here or select a known one from the left" />'+
'  <input type="hidden" id="'+this.base+'-m-tag-value-'+i+'-'+id+'" name="tag_value_'+i+'" value="" />'+
'</div>';

		var entry = this.messages[id];
		var html =
'<div id="'+this.base+'-m-rdlg-'+id+'" class="el-l-m-rdlg el-l-m-dlg-hdn">'+
'  <div id="'+this.base+'-m-rdlg-'+id+'-info" style="color:maroon; position:relative; left:-10px; top:-15px;">Compose reply. Note the total limit of <b>25 MB</b> for attachments.</div>'+
'  <div style="float:left;">'+
'    <form id="elog-form-post-'+id+'" enctype="multipart/form-data" action="../logbook/NewFFEntry4portalJSON.php" method="post">'+
'      <input type="hidden" name="id" value="'+this.parent.exp_id+'" />'+
'      <input type="hidden" name="scope" value="message" />'+
'      <input type="hidden" name="message_id" value="'+id+'" />'+
'      <input type="hidden" name="run_id" value="" />'+
'      <input type="hidden" name="shift_id" value="" />'+
'      <input type="hidden" name="MAX_FILE_SIZE" value="25000000" />'+
'      <input type="hidden" name="num_tags" value="0" />'+
'      <input type="hidden" name="onsuccess" value="" />'+
'      <input type="hidden" name="relevance_time" value="" />'+
'      <textarea name="message_text" rows="12" cols="64" style="padding:4px;" title="the first line of the message body will be used as its subject" ></textarea>'+
'      <div style="margin-top: 10px;">'+
'        <div style="float:left;">'+ 
'          <div style="font-weight:bold;">Author:</div>'+
'          <input type="text" name="author_account" value="'+this.parent.author+'" size=32 style="padding:2px; margin-top:5px;" />'+
'        </div>'+
'        <div style="float:left; margin-left:20px;">'+ 
'          <div style="font-weight:bold;">Attachments:</div>'+
'          <div id="'+this.base+'-reply-as-'+id+'" style="margin-top:5px;">'+
'            <div>'+
'              <input type="file" name="file2attach_0" onchange="'+this.address+'.live_reply_add_attachment('+id+')" />'+
'              <input type="hidden" name="file2attach_0" value=""/ >'+
'            </div>'+
'          </div>'+
'        </div>'+
'        <div style="clear:both;"></div>'+
'      </div>'+
'    </form>'+
'  </div>'+
'  <div style="float:right; margin-left:5px;" class="s-b-con"><button id="'+this.base+'-m-re-c-'+id+'" onclick="'+this.address+'.live_message_reply_cancel('+id+');">cancel</button></div>'+
'  <div style="float:right; margin-left:5px;" class="s-b-con"><button id="'+this.base+'-m-re-s-'+id+'" onclick="'+this.address+'.live_message_reply_submit('+id+');">submit</button></div>'+
'  <div style="clear:both;"></div>'+
'</div>'+
'<div id="'+this.base+'-m-xtdlg-'+id+'" class="el-l-m-xtdlg el-l-m-dlg-hdn">'+
'  <div id="'+this.base+'-m-xtdlg-'+id+'-info" style="color:maroon; position:relative; left:-10px; top:-15px;">Select existing tags or define new tags to be added.</div>'+
'  <div style="float:left;">'+
'    <form id="elog-form-extend-tags-'+id+'" enctype="multipart/form-data" action="../logbook/ExtendFFEntry4portalJSON.php" method="post">'+
'      <input type="hidden" name="message_id" value="'+id+'" />'+
'      <input type="hidden" name="MAX_FILE_SIZE" value="25000000" />'+
'      <input type="hidden" name="num_tags" value="'+num_tags+'" />'+
'      <input type="hidden" name="onsuccess" value="" />'+
'      <div>'+
'        <div>'+tags_html+'</div>'+
'      </div>'+
'    </form>'+
'  </div>'+
'  <div style="float:right; margin-left:5px;" class="s-b-con"><button id="'+this.base+'-m-xt-c-'+id+'" onclick="'+this.address+'.live_message_extend_cancel('+id+', true);">cancel</button></div>'+
'  <div style="float:right; margin-left:5px;" class="s-b-con"><button id="'+this.base+'-m-xt-s-'+id+'" onclick="'+this.address+'.live_message_extend_submit('+id+', true);">submit</button></div>'+
'  <div style="clear:both;"></div>'+
'</div>'+
'<div id="'+this.base+'-m-xadlg-'+id+'" class="el-l-m-xadlg el-l-m-dlg-hdn">'+
'  <div id="'+this.base+'-m-xadlg-'+id+'-info" style="color:maroon; position:relative; left:-10px; top:-15px;">Select attachments to upload. Note the total limit of <b>25 MB</b>.</div>'+
'  <div style="float:left;">'+
'    <form id="elog-form-extend-attachments-'+id+'" enctype="multipart/form-data" action="../logbook/ExtendFFEntry4portalJSON.php" method="post">'+
'      <input type="hidden" name="message_id" value="'+id+'" />'+
'      <input type="hidden" name="MAX_FILE_SIZE" value="25000000" />'+
'      <input type="hidden" name="num_tags" value="0" />'+
'      <input type="hidden" name="onsuccess" value="" />'+
'      <div>'+
'        <div id="'+this.base+'-extend-as-'+id+'">'+
'          <div>'+
'            <input type="file" name="file2attach_0" onchange="'+this.address+'.live_extend_add_attachment('+id+')" />'+
'            <input type="hidden" name="file2attach_0" value=""/ >'+
'          </div>'+
'        </div>'+
'      </div>'+
'    </form>'+
'  </div>'+
'  <div style="float:right; margin-left:5px;" class="s-b-con"><button id="'+this.base+'-m-xa-c-'+id+'" onclick="'+this.address+'.live_message_extend_cancel('+id+', false);">cancel</button></div>'+
'  <div style="float:right; margin-left:5px;" class="s-b-con"><button id="'+this.base+'-m-xa-s-'+id+'" onclick="'+this.address+'.live_message_extend_submit('+id+', false);">submit</button></div>'+
'  <div style="clear:both;"></div>'+
'</div>';
		if(this.parent.editor) {
			html +=
'<div id="'+this.base+'-m-edlg-'+id+'" class="el-l-m-edlg el-l-m-dlg-hdn">'+
'  <div style="font-size:90%; text-decoration:underline; position:relative; left:-10px; top:-15px;">E d i t &nbsp; m e s s a g e</div>'+
'  <div style="float:left;">'+
'    <form id="elog-form-edit-'+id+'" enctype="multipart/form-data" action="../logbook/UpdateFFEntry4portal.php" method="post">'+
'      <input type="hidden" name="id" value="'+id+'" />'+
'      <input type="hidden" name="content_type" value="TEXT" />'+
'      <input type="hidden" name="onsuccess" value="" />'+
'      <textarea name="content" rows="12" cols="64" style="padding:4px;" title="the first line of the message body will be used as its subject" >'+entry.content+'</textarea>'+
'      <div style="font-weight:bold; margin-top: 10px;">Extra attachments:</div>'+
'      <div id="'+this.base+'-edit-as-'+id+'" style="margin-top:5px;">'+
'        <div>'+
'          <input type="file" name="file2attach_0" onchange="'+this.address+'.live_edit_add_attachment('+id+')" />'+
'          <input type="hidden" name="file2attach_0" value=""/ >'+
'        </div>'+
'      </div>'+
'    </form>'+
'  </div>'+
'  <div style="float:right; margin-left:5px;" class="s-b-con"><button class="el-l-m-ed-c" id="'+this.base+'-m-ed-c-'+id+'" onclick="'+this.address+'.live_message_edit_cancel('+id+');">cancel</button></div>'+
'  <div style="float:right; margin-left:5px;" class="s-b-con"><button class="el-l-m-ed-s" id="'+this.base+'-m-ed-s-'+id+'" onclick="'+this.address+'.live_message_edit_submit('+id+');">submit</button></div>'+
'  <div style="clear:both;"></div>'+
'</div>'+
'<div id="'+this.base+'-m-mdlg-'+id+'" class="el-l-m-mdlg el-l-m-dlg-hdn">'+
'  <div style="float:right; margin-left:5px;" class="s-b-con"><button class="el-l-m-mv-c" id="'+this.base+'-m-mv-c-'+id+'" onclick="'+this.address+'.live_message_move_cancel('+id+');">cancel</button></div>'+
'  <div style="float:right; margin-left:5px;" class="s-b-con"><button class="el-l-m-mv-s" id="'+this.base+'-m-mv-s-'+id+'" onclick="'+this.address+'.live_message_move_submit('+id+');">submit</button></div>'+
'  <div style="clear:both;"></div>'+
''+
'  Here be the dialog allowing to move the message into some other context and/or time.'+
'  Right now we create it automatically for each message after loading it.'+
'  We should probably optimize things by creating it at first use.'+
'  That way we will reduce the size of DOM'+
'</div>';
		}

		dlgs.html(html);

		$('#'+this.base+'-m-xt-s-'+id).button();
		$('#'+this.base+'-m-xt-c-'+id).button();
		$('#'+this.base+'-m-xa-s-'+id).button();
		$('#'+this.base+'-m-xa-c-'+id).button();

		$('#'+this.base+'-m-re-s-'+id).button();
		$('#'+this.base+'-m-re-c-'+id).button();

		if(this.parent.editor) {
			$('#'+this.base+'-m-ed-s-'+id).button();
			$('#'+this.base+'-m-ed-c-'+id).button();
			$('#'+this.base+'-m-mv-e-'+id).button();
			$('#'+this.base+'-m-mv-c-'+id).button();
		}
		$('#'+this.base+'-m-tags-library-0-'+id).change(function(ev) {
			var selectedIndex = $('#'+that.base+'-m-tags-library-0-'+id).attr('selectedIndex');
			if( selectedIndex == 0 ) return;
			$('#'+that.base+'-m-tag-name-0-'+id    ).val($('#'+that.base+'-m-tags-library-0-'+id).val());
			$('#'+that.base+'-m-tags-library-0-'+id).attr('selectedIndex', 0);
		});
		$('#'+this.base+'-m-tags-library-1-'+id).change(function(ev) {
			var selectedIndex = $('#'+that.base+'-m-tags-library-1-'+id).attr('selectedIndex');
			if( selectedIndex == 0 ) return;
			$('#'+that.base+'-m-tag-name-1-'+id    ).val($('#'+that.base+'-m-tags-library-1-'+id).val());
			$('#'+that.base+'-m-tags-library-1-'+id).attr('selectedIndex', 0);
		});
		$('#'+this.base+'-m-tags-library-2-'+id).change(function(ev) {
			var selectedIndex = $('#'+that.base+'-m-tags-library-2-'+id).attr('selectedIndex');
			if( selectedIndex == 0 ) return;
			$('#'+that.base+'-m-tag-name-2-'+id    ).val($('#'+that.base+'-m-tags-library-2-'+id).val());
			$('#'+that.base+'-m-tags-library-2-'+id).attr('selectedIndex', 0);
		});
	};

	this.live_enable_message_buttons = function(id, on) {
		var state = on ? 'enable' : 'disable';
		$('#'+this.base+'-m-xt-'+id).button(state);
		$('#'+this.base+'-m-xa-'+id).button(state);
		$('#'+this.base+'-m-re-'+id).button(state);
		$('#'+this.base+'-m-ed-'+id).button(state);
		$('#'+this.base+'-m-mv-'+id).button(state);
	};
	this.live_message_extend = function(id, is_child, add_tags_vs_attachments) {
	    that.create_live_message_dialogs(id, is_child);
	    if( add_tags_vs_attachments ) $('#'+this.base+'-m-xtdlg-'+id).removeClass('el-l-m-dlg-hdn').addClass('el-l-m-dlg-vis');
	    else                          $('#'+this.base+'-m-xadlg-'+id).removeClass('el-l-m-dlg-hdn').addClass('el-l-m-dlg-vis');
		that.live_enable_message_buttons(id, false);
	};
	this.live_message_reply = function(id, is_child) {
	    that.create_live_message_dialogs(id, is_child);
		$('#'+this.base+'-m-rdlg-'+id).removeClass('el-l-m-dlg-hdn').addClass('el-l-m-dlg-vis');
		that.live_enable_message_buttons(id, false);
		$('#'+this.base+'-m-rdlg-'+id+' textarea').focus();
	};
	this.live_message_edit = function(id, is_child) {
	    that.create_live_message_dialogs(id, is_child);
		$('#'+this.base+'-m-edlg-'+id).removeClass('el-l-m-dlg-hdn').addClass('el-l-m-dlg-vis');
		that.live_enable_message_buttons(id, false);
		$('#'+this.base+'-m-edlg-'+id+' textarea').focus();
	};
	this.live_message_move = function(id, is_child) {
	    that.create_live_message_dialogs(id, is_child);
		$('#'+this.base+'-m-mdlg-'+id).removeClass('el-l-m-dlg-hdn').addClass('el-l-m-dlg-vis');
		that.live_enable_message_buttons(id, false);
	};

	this.live_message_reply_cancel = function(id) {
		$('#'+this.base+'-m-rdlg-'+id).removeClass('el-l-m-dlg-vis').addClass('el-l-m-dlg-hdn');
		that.live_enable_message_buttons(id, true);
	};
	this.live_message_edit_cancel = function(id) {
		$('#'+this.base+'-m-edlg-'+id).removeClass('el-l-m-dlg-vis').addClass('el-l-m-dlg-hdn');
		that.live_enable_message_buttons(id, true);
	};
	this.live_message_move_cancel = function(id) {
		$('#'+this.base+'-m-mdlg-'+id).removeClass('el-l-m-dlg-vis').addClass('el-l-m-dlg-hdn');
		that.live_enable_message_buttons(id, true);
	};

	this.live_message_reply_submit = function(id) {

		var urlbase = window.location.href;
		var idx = urlbase.indexOf( '&' );
		if( idx > 0 ) urlbase = urlbase.substring( 0, idx );

		// Make sure there is anything to submit. Do not allow submitting
		// an empty message.
		//
		if( $('#elog-form-post-'+id+' textarea[name="message_text"]').val() == '' ) {
			this.parent.report_error('Can not post the empty message. Please put some text into the message box.' );
			return;
		}

		var old_info = $('#'+this.base+'-m-rdlg-'+id+'-info').html();
		$('#'+this.base+'-m-rdlg-'+id+'-info').html('Posting. Please wait...');
		$('#'+this.base+'-m-re-c-'+id).button('disable');
		$('#'+this.base+'-m-re-s-'+id).button('disable');

		// Use JQuery AJAX Form plug-in to post the reply w/o reloading
		// the current page.
		//
		$('#elog-form-post-'+id).ajaxSubmit({
			success: function(data) {
				$('#elog-form-post-'+id+' textarea[name="message_text"]').val('');
				$('#elog-form-post-'+id+' input[name="author_account"]').val(that.parent.author);
				$('#'+that.base+'-reply-as-'+id).html(
'<div>'+
'  <input type="file" name="file2attach_0" onchange="'+that.address+'.live_reply_add_attachment('+id+')" />'+
'  <input type="hidden" name="file2attach_0" value=""/ >'+
'</div>'
				);
				if( data.Status != 'success' ) { that.parent.report_error( data.Message ); return; }
				var entry = data.Entry;

				// Find the parent message and add a new child to it.
				//
				// TODO: Should we find a way to serialize the JSON object into a string?
				//       A problem is that the children array updated below is also used
				//       from other locations of the code. And in those other locations
				//       elements of the children array are treated as strings, hence they're
				//       converted into valid JavaScript objects before passing them into
				//       the live_child2html() function
				//
				//       Another possibility would be to use typeof inside the code
				//       which uses the array of children to see if this is the string
				//       and retranslate it into JSON automatically, or do it immediatellty
				//       after loading.
				//
				that.messages[id].children.push( data );

				// Then display the new child on the page and make it visible
				//
				var html = that.live_child2html(entry, that.messages[id].thread_idx);
				$('#'+that.base+'-m-c-'+id).prepend(html);

				$('.el-l-m-re').button();
				$('.el-l-m-ed').button();
				$('.el-l-m-mv').button();

				that.toggle_child(entry.id);

				that.live_message_reply_cancel(id);
			},
			complete: function() {
				$('#'+that.base+'-m-rdlg-'+id+'-info').html( old_info );
				$('#'+that.base+'-m-re-c-'+id).button('enable');
				$('#'+that.base+'-m-re-s-'+id).button('enable');
			},
			dataType: 'json'
		});
	};

	this.live_extend_add_attachment = function(id) {
		var num = $('#'+this.base+'-extend-as-'+id+' > div').size();
		$('#'+this.base+'-extend-as-'+id).append(
'  <div>'+
'    <input type="file" name="file2attach_'+num+'" onchange="'+this.address+'.live_extend_add_attachment('+id+')" />'+
'    <input type="hidden" name="file2attach_'+num+'" value=""/ >'+
'  </div>'
		);
	};
	this.live_message_extend_cancel = function(id, add_tags_vs_attachments ) {
		if( add_tags_vs_attachments ) $('#'+this.base+'-m-xtdlg-'+id).removeClass('el-l-m-dlg-vis').addClass('el-l-m-dlg-hdn');
		else                          $('#'+this.base+'-m-xadlg-'+id).removeClass('el-l-m-dlg-vis').addClass('el-l-m-dlg-hdn');
		that.live_enable_message_buttons(id, true);
	};
	this.live_message_extend_submit = function(id, add_tags_vs_attachments) {

		var urlbase = window.location.href;
		var idx = urlbase.indexOf( '&' );
		if( idx > 0 ) urlbase = urlbase.substring( 0, idx );

		// Use JQuery AJAX Form plug-in to post the reply w/o reloading
		// the current page.
		//
		if( add_tags_vs_attachments ) {

			var old_info = $('#'+this.base+'-m-xtdlg-'+id+'-info').html();
			$('#'+this.base+'-m-xtdlg-'+id+'-info').html('Submitting. Please wait...');
			$('#'+this.base+'-m-xt-c-'+id).button('disable');
			$('#'+this.base+'-m-xt-s-'+id).button('disable');

			$('#elog-form-extend-tags-'+id).ajaxSubmit({
				success: function(data) {
					if( data.Status != 'success' ) { that.parent.report_error( data.Message ); return; }
					var tags = data.Extended.tags;
					for( var i = 0; i < tags.length; i++ )
						that.messages[id].tags.unshift(tags[i]);

					that.expand_message(that.messages[id].thread_idx, true);
				},
				complete: function() {
					$('#'+that.base+'-m-xtdlg-'+id+'-info').html( old_info );
					$('#'+that.base+'-m-xt-c-'+id).button('enable');
					$('#'+that.base+'-m-xt-s-'+id).button('enable');
				},
				dataType: 'json'
			});
		} else {

			var old_info = $('#'+this.base+'-m-xadlg-'+id+'-info').html();
			$('#'+this.base+'-m-xadlg-'+id+'-info').html('Uploading. Please wait...');
			$('#'+this.base+'-m-xa-c-'+id).button('disable');
			$('#'+this.base+'-m-xa-s-'+id).button('disable');

			$('#elog-form-extend-attachments-'+id).ajaxSubmit({
				success: function(data) {
					$('#'+that.base+'-extend-as-'+id).html(
'<div>'+
'  <input type="file" name="file2attach_0" onchange="'+that.address+'.live_extend_add_attachment('+id+')" />'+
'  <input type="hidden" name="file2attach_0" value=""/ >'+
'</div>'
					);
					if( data.Status != 'success' ) { that.parent.report_error( data.Message ); return; }
					var attachments = data.Extended.attachments;
					for( var i = 0; i < attachments.length; i++ )
						that.messages[id].attachments.unshift(attachments[i]);

					that.expand_message(that.messages[id].thread_idx, true);
				},
				complete: function() {
					$('#'+that.base+'-m-xadlg-'+id+'-info').html( old_info );
					$('#'+that.base+'-m-xa-c-'+id).button('enable');
					$('#'+that.base+'-m-xa-s-'+id).button('enable');
				},
				dataType: 'json'
			});
		}
	};



	this.live_message_edit_submit = function(id) {
		var urlbase = window.location.href;
		var idx = urlbase.indexOf( '&' );
		if( idx > 0 ) urlbase = urlbase.substring( 0, idx );
		$('#elog-form-edit-'+id+' input[name="onsuccess"]').val(urlbase+'&app=elog');

		$('#elog-form-edit-'+id).trigger( 'submit' );
	};
	this.live_message_move_submit = function(id) {
		$('#'+this.base+'-m-mdlg-'+id).removeClass('el-l-m-dlg-vis').addClass('el-l-m-dlg-hdn');
		that.live_enable_message_buttons(id, true);
	};

	this.live_reply_add_attachment = function(id) {
		var num = $('#'+this.base+'-reply-as-'+id+' > div').size();
		$('#'+this.base+'-reply-as-'+id).append(
'  <div>'+
'    <input type="file" name="file2attach_'+num+'" onchange="'+this.address+'.live_reply_add_attachment('+id+')" />'+
'    <input type="hidden" name="file2attach_'+num+'" value=""/ >'+
'  </div>'
		);
	};
	this.live_edit_add_attachment = function(id) {
		var num = $('#'+this.base+'-edit-as-'+id+' > div').size();
		$('#'+this.base+'-edit-as-'+id).append(
'  <div>'+
'    <input type="file" name="file2attach_'+num+'" onchange="'+this.address+'.live_edit_add_attachment('+id+')" />'+
'    <input type="hidden" name="file2attach_'+num+'" value=""/ >'+
'  </div>'
		);
	};


	this.live_run_reply_cancel = function(id) {
		$('#'+this.base+'-r-rdlg-'+id).removeClass('el-l-r-dlg-vis').addClass('el-l-r-dlg-hdn');
		that.live_enable_run_buttons(id, true);
	};
	this.live_run_reply_submit = function(idx) {

		var entry = this.threads[idx];
		var id    = entry.id;

		var urlbase = window.location.href;
		var urlbase_idx = urlbase.indexOf( '&' );
		if( urlbase_idx > 0 ) urlbase = urlbase.substring( 0, urlbase_idx );

		// Make sure there is anything to submit. Do not allow submitting
		// an empty message.
		//
		if( $('#elog-form-post-'+id+' textarea[name="message_text"]').val() == '' ) {
			this.parent.report_error('Can not post the empty message. Please put some text into the message box.');
			return;
		}

		// Use JQuery AJAX Form plug-in to post the reply w/o reloading
		// the current page.
		//
		var old_info = $('#'+this.base+'-m-rdlg-'+id+'-info').html();
		$('#'+this.base+'-r-rdlg-'+id+'-info').html('Posting. Please wait...');
		$('#'+this.base+'-r-re-c-'+id).button('disable');
		$('#'+this.base+'-r-re-s-'+id).button('disable');

		$('#elog-form-post-'+id).ajaxSubmit({
			success: function(data) {
				$('#elog-form-post-'+id+' textarea[name="message_text"]').val('');
				$('#elog-form-post-'+id+' input[name="author_account"]').val(that.parent.author);
				$('#'+that.base+'-reply-as-'+id).html(
'<div>'+
'  <input type="file" name="file2attach_0" onchange="'+that.address+'.live_reply_add_attachment('+id+')" />'+
'  <input type="hidden" name="file2attach_0" value=""/ >'+
'</div>'
				);
				if( data.Status != 'success' ) { that.parent.report_error( data.Message ); return; }
				that.live_run_reply_cancel(id);

				// Close and reopen the panel for the run to refresh its contents, so that
				// the new message would instanteneously appear here.
				//
				that.expand_message( idx, false );
				that.expand_message( idx, true );
			},
			complete: function() {
				$('#'+that.base+'-r-rdlg-'+id+'-info').html( old_info );
				$('#'+that.base+'-r-re-c-'+id).button('enable');
				$('#'+that.base+'-r-re-s-'+id).button('enable');
			},
			dataType: 'json'
		});
	};

	this.create_live_run_dialogs = function(idx) {
		var entry = this.threads[idx];
		var id    = entry.id;
		var dlgs  = $('#'+this.base+'-r-dlgs-'+id);
		if(dlgs.html() != '') return;
		var html =
'<div id="'+this.base+'-r-rdlg-'+id+'" class="el-l-r-rdlg el-l-r-dlg-hdn">'+
//'  <div style="font-size:90%; text-decoration:underline; position:relative; left:-10px; top:-15px;">C o m p o s e &nbsp; m e s s a g e</div>'+
'  <div id="'+this.base+'-r-rdlg-'+id+'-info" style="color:maroon; position:relative; left:-10px; top:-15px;">Compose message. Note the total limit of <b>25 MB</b> for attachments.</div>'+
'  <div style="float:left;">'+
'    <form id="elog-form-post-'+id+'" enctype="multipart/form-data" action="../logbook/NewFFEntry4portalJSON.php" method="post">'+
'      <input type="hidden" name="id" value="'+this.parent.exp_id+'" />'+
'      <input type="hidden" name="scope" value="run" />'+
'      <input type="hidden" name="message_id" value="" />'+
'      <input type="hidden" name="run_id" value="'+entry.run_id+'" />'+
'      <input type="hidden" name="shift_id" value="" />'+
'      <input type="hidden" name="MAX_FILE_SIZE" value="25000000" />'+
'      <input type="hidden" name="num_tags" value="0" />'+
'      <input type="hidden" name="onsuccess" value="" />'+
'      <input type="hidden" name="relevance_time" value="" />'+
'      <textarea name="message_text" rows="12" cols="64" style="padding:4px;" title="the first line of the message body will be used as its subject" ></textarea>'+
'      <div style="margin-top: 10px;">'+
'        <div style="float:left;">'+ 
'          <div style="font-weight:bold;">Author:</div>'+
'          <input type="text" name="author_account" value="'+this.parent.author+'" size=32 style="padding:2px; margin-top:5px;" />'+
'        </div>'+
'        <div style="float:left; margin-left:20px;">'+ 
'          <div style="font-weight:bold;">Attachments:</div>'+
'          <div id="'+this.base+'-reply-as-'+id+'" style="margin-top:5px;">'+
'            <div>'+
'              <input type="file" name="file2attach_0" onchange="'+this.address+'.live_reply_add_attachment('+id+')" />'+
'              <input type="hidden" name="file2attach_0" value=""/ >'+
'            </div>'+
'          </div>'+
'        </div>'+
'        <div style="clear:both;"></div>'+
'      </div>'+
'    </form>'+
'  </div>'+
'  <div style="float:right; margin-left:5px;" class="s-b-con"><button id="'+this.base+'-r-re-c-'+id+'" onclick="'+this.address+'.live_run_reply_cancel('+id+');">cancel</button></div>'+
'  <div style="float:right; margin-left:5px;" class="s-b-con"><button id="'+this.base+'-r-re-s-'+id+'" onclick="'+this.address+'.live_run_reply_submit('+idx+');">submit</button></div>'+
'  <div style="clear:both;"></div>'+
'</div>';

		dlgs.html(html);

		$('#'+this.base+'-r-re-s-'+id).button();
		$('#'+this.base+'-r-re-c-'+id).button();

	};
	this.live_enable_run_buttons = function(id, on) {
		var state = on ? 'enable' : 'disable';
		$('#'+this.base+'-r-re-'+id).button(state);
	};
	this.live_run_reply = function(idx) {
	    that.create_live_run_dialogs(idx);
		var entry = this.threads[idx];
		var id    = entry.id;
		$('#'+this.base+'-r-rdlg-'+id).removeClass('el-l-r-dlg-hdn').addClass('el-l-r-dlg-vis');
		that.live_enable_run_buttons(id, false);
		$('#'+this.base+'-r-rdlg-'+id+' textarea').focus();
	};

	this.live_thread2html = function(thread_idx) {
		var html = '';
		var entry = that.threads[thread_idx];
		if(!entry.is_run) {
			that.messages[entry.id] = entry;
			html +=
'  <div class="el-l-m-hdr" id="'+this.base+'-m-hdr-'+thread_idx+'" onclick="'+this.address+'.toggle_message('+thread_idx+');">'+
'    <div style="float:left;"><span class="toggler ui-icon ui-icon-triangle-1-e el-l-m-tgl" id="'+this.base+'-m-tgl-'+entry.id+'"></span></div>'+
'    <div style="float:left;" class="el-r-m-time">'+entry.relevance_time+'</div>'+
'    <div style="float:left;" class="el-l-m-author">'+entry.author+'</div>'+
'    <div style="float:left; margin-left:10px;" class="el-l-m-subj">'+entry.subject+'</div>'+id_sign(entry)+run_sign(entry.run_num)+message_signs(entry)+
'    <div style="clear:both;"></div>'+
'  </div>'+
'  <div class="el-l-m-con el-l-m-hdn" id="'+this.base+'-m-con-'+entry.id+'"></div>';

			// The message contents container will be dynamically initialized at its
			// first use (call to the toggle_message() function.
			//
			;

		} else {
        	that.runs.push(thread_idx);
			html +=
'  <div class="el-l-r-hdr" id="'+this.base+'-m-hdr-'+thread_idx+'" onclick="'+this.address+'.toggle_run('+thread_idx+');">'+
'    <div style="float:left;"><span class="toggler ui-icon ui-icon-triangle-1-e el-l-r-tgl" id="'+this.base+'-r-tgl-'+entry.id+'"></span></div>'+
'    <div style="float:left;" class="el-r-m-time">'+entry.relevance_time+'</div>'+
'    <div style="float:left;" class="el-l-m-author">'+entry.author+'</div>'+
'    <div style="float:left; margin-left:10px;" class="el-l-m-subj">'+entry.subject+'</div>'+
'    <div style="clear:both;"></div>'+
'  </div>'+
'  <div class="el-l-r-con el-l-r-hdn" id="'+this.base+'-r-con-'+entry.id+'"></div>';

			// The run contents container will be dynamically initialized at its
			// first use (call to the toggle_run() function.
			//
			;
		}
		return html;
	};
/*
	this.live_child2html = function(entry) {
		that.messages[entry.id] = entry;
		var html =
'<div class="el-l-c-hdr" onclick="'+this.address+'.toggle_child('+entry.id+');">'+
'  <div style="float:left;"><span class="toggler ui-icon ui-icon-triangle-1-e el-l-c-tgl" id="'+this.base+'-c-tgl-'+entry.id+'"></span></div>'+
'  <div style="float:left;" class="el-l-c-time">'+entry.relevance_time+'</div>'+
'  <div style="float:left;" class="el-l-c-author">'+entry.author+'</div>'+
'  <div style="float:left; margin-left:10px;" class="el-l-c-subj">'+entry.subject+'</div>'+id_sign(entry)+message_signs(entry)+
'  <div style="clear:both;"></div>'+
'</div>'+
'<div class="el-l-c-con el-l-c-hdn" id="'+this.base+'-c-con-'+entry.id+'">'+
'  <div class="el-l-c-body">'+
'    <div style="float:left; font-size:12px; width:100%; overflow:auto;">'+entry.html1+'</div>';
		if(this.parent.editor)
			html +=
'    <div style="float:right;" class="s-b-con"><button class="el-l-m-ed"  id="'+this.base+'-m-ed-'+entry.id+'"  onclick="'+this.address+'.live_message_edit('+entry.id+',true);">edit</button></div>';
		html +=
'    <div style="float:right;" class="s-b-con"><button class="el-l-m-re" id="'+this.base+'-m-re-'+entry.id+'" onclick="'+this.address+'.live_message_reply('+entry.id+',true);">reply</button></div>'+
'    <div style="clear:both;"></div>'+
'      <div id="'+this.base+'-m-dlgs-'+entry.id+'"></div>'+
'  </div>';
		that.attachments[entry.id] = entry.attachments;

		var attachments_html = '';
		for( var i in entry.attachments) {
			var a = entry.attachments[i];
			that.attachments_loader[a.id] = {loaded: false, descr: a};
			attachments_html +=
'    <div style="float:left;" class="el-l-a">'+
'      <div style="float:left;">'+
'        <span class="toggler ui-icon ui-icon-triangle-1-e el-l-a-tgl" id="'+this.base+'-a-tgl-'+a.id+'" onclick="'+this.address+'.toggle_attachment('+a.id+');"></span>'+
'      </div>'+
'      <div style="float:left;" class="el-l-a-dsc"><a class="link" href="../logbook/attachments/'+a.id+'/preview"  target="_blank">'+a.description+'</a></div>'+
'      <div style="float:left; margin-left:10px;" class="el-l-a-info">( type: <b>'+a.type+'</b> size: <b>'+a.size+'</b> )</div>'+
'      <div style="clear:both;"></div>'+
'      <div class="el-l-a-con el-l-a-hdn" id="'+this.base+'-a-con-'+a.id+'">'+
'      </div>'+
'    </div>';
		}
		if(attachments_html) html +=
'  <div class="el-l-m-as">'+attachments_html+
'    <div style="clear:both;"></div>'+
'  </div>';
		html +=
'  <div id="'+this.base+'-m-c-'+entry.id+'">';

		var children = entry.children;
		for(var i in children) html += that.live_child2html(eval("("+children[i]+")"));

		html +=
'  </div>'+
'</div>';
		return html;
	};
*/
	this.live_child2html = function(entry, thread_idx) {
		that.messages[entry.id] = entry;
		var html =
'<div class="el-l-c-hdr" onclick="'+this.address+'.toggle_child('+entry.id+');">'+
'  <div style="float:left;"><span class="toggler ui-icon ui-icon-triangle-1-e el-l-c-tgl" id="'+this.base+'-c-tgl-'+entry.id+'"></span></div>'+
'  <div style="float:left;" class="el-l-c-time">'+entry.relevance_time+'</div>'+
'  <div style="float:left;" class="el-l-c-author">'+entry.author+'</div>'+
'  <div style="float:left; margin-left:10px;" class="el-l-c-subj">'+entry.subject+'</div>'+id_sign(entry)+message_signs(entry)+
'  <div style="clear:both;"></div>'+
'</div>'+
'<div class="el-l-c-con el-l-c-hdn" id="'+this.base+'-c-con-'+entry.id+'">'+
'  <div class="el-l-c-body">'+
'    <div style="float:left; font-size:12px; width:100%; overflow:auto;">'+entry.html1+'</div>';
		if(this.parent.editor)
			html +=
'    <div style="float:right;" class="s-b-con"><button class="el-l-m-ed"  id="'+this.base+'-m-ed-'+entry.id+'"  onclick="'+this.address+'.live_message_edit('+entry.id+',true);">edit</button></div>';
		html +=
'    <div style="float:right;" class="s-b-con"><button class="el-l-m-re" id="'+this.base+'-m-re-'+entry.id+'" onclick="'+this.address+'.live_message_reply('+entry.id+',true);">reply</button></div>'+
'    <div style="clear:both;"></div>'+
'      <div id="'+this.base+'-m-dlgs-'+entry.id+'"></div>'+
'  </div>';
		that.attachments[entry.id] = entry.attachments;

		var attachments_html = '';
		for( var i in entry.attachments) {
			var a = entry.attachments[i];
			that.attachments_loader[a.id] = {loaded: false, descr: a};
			attachments_html +=
'    <div style="float:left;" class="el-l-a">'+
'      <div style="float:left;">'+
'        <span class="toggler ui-icon ui-icon-triangle-1-e el-l-a-tgl" id="'+this.base+'-a-tgl-'+a.id+'" onclick="'+this.address+'.toggle_attachment('+a.id+');"></span>'+
'      </div>'+
'      <div style="float:left;" class="el-l-a-dsc"><a class="link" href="../logbook/attachments/'+a.id+'/preview"  target="_blank">'+a.description+'</a></div>'+
'      <div style="float:left; margin-left:10px;" class="el-l-a-info">( type: <b>'+a.type+'</b> size: <b>'+a.size+'</b> )</div>'+
'      <div style="clear:both;"></div>'+
'      <div class="el-l-a-con el-l-a-hdn" id="'+this.base+'-a-con-'+a.id+'">'+
'      </div>'+
'    </div>';
		}
		if(attachments_html) html +=
'  <div class="el-l-m-as">'+attachments_html+
'    <div style="clear:both;"></div>'+
'  </div>';
		html +=
'  <div id="'+this.base+'-m-c-'+entry.id+'">';

		var children = entry.children;
		for(var i in children) html += that.live_child2html(eval("("+children[i]+")"), thread_idx);

		html +=
'  </div>'+
'</div>';
		return html;
	};
	function message_signs(entry) {
		var html = '<div style="float:right; margin-right:10px;">';
		for(var i=0; i < entry.attachments.length; i++) html += '<img src="../logbook/images/attachment.png" height="18" />';
		for(var i=0; i < entry.tags.length; i++) html += '&nbsp;<sup><b>T</b></sup>';
		if(entry.children.length) html +=  '&nbsp;<sup><b>&crarr;</b></sup>';
		html += '</div>';
		return html;
	}
	function id_sign(entry) {
		var html = '<div style="float:right; margin-left:10px; margin-top:2px; padding-left:2px; padding-right:2px; background-color:#ffffff;">'+entry.id+'</div>';
		return html;
	}

	function run_sign(num) {
		if( num ) {
			var html = '<div style="float:right; margin-right:10px;"><sup><b>run: '+num+'</b></sup></div>';
			return html;
		}
		return '';
	}
}









