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
	this.rrange  = null;
	this.min_run = null;
	this.max_run = null;
	this.shifts  = new Array();
	this.runs    = new Array();
	this.editor  = false;

	/* The context for v-menu items
	 */
	var context2_default = {
		'recent' : '20',
		'post'   : 'experiment',
		'search' : 'simple',
		'browse' : '',
		'shifts' : '',
		'runs'   : '',
		'subscribe' : ''
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
			if( just_initialized || (prev_context1 != this.context1) || (prev_context2 != this.context2)) {
				this.live_current_selected_range = this.context2;
				this.live_dim_all_highlights();
				this.live_message_viewer.reload(live_selected_runs(), this.live_selected_range());
			}
		} else if(this.context1 == 'post') {

			$('#elog-form-post input[name="scope"]').val(this.context1);
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

		} else if(this.context1 == 'search') {
			if(this.context2 == 'simple') {
				$('#elog-form-search input').attr('disabled', 'disabled');
				$('#elog-form-search select').attr('disabled', 'disabled');
				$('#elog-form-search input[name="text2search"]').removeAttr('disabled');
			} else {
				$('#elog-form-search input').removeAttr('disabled');
				$('#elog-form-search select').removeAttr('disabled');
			}
		}
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
	
			/* Validate the form and initialize hidden fields */

			var urlbase = window.location.href;
			var idx = urlbase.indexOf( '&' );
			if( idx > 0 ) urlbase = urlbase.substring( 0, idx );
			$('#elog-form-post input[name="onsuccess"]').val(urlbase+'&page1=elog&page2=post');

			if( that.context2 == 'run' ) {

				if( that.min_run == null ) {
					$('#el-p-runnum-error').text('the experiment has not taken any runs yet');
					return;
				}
				var run = $('#el-p-runnum').val();
				if(( run < that.min_run ) || ( run > that.max_run )) {
					$('#el-p-runnum-error').text('the run number is out of allowed range: '+that.min_run+'-'+that.max_run);
					return;
				}
				$('#elog-form-post input[name="run_id"]').val(that.runs[run]);

			} else if( that.context2 == 'shift' ) {

				var shift = $('#el-p-shift').val();
				$('#elog-form-post input[name="shift_id"]').val(that.shifts[shift]);
			}
			if( post_selected_relevance() == 'past' ) {

				/* TODO: Check the syntax of the timestamp using regular expression
				 *       before submitting the request. The server side script will also
				 *       check its validity (applicability).
				 */
				var relevance_time = $('#el-p-datepicker').val()+' '+$('#el-p-time').val();
				$('#elog-form-post input[name="relevance_time"]').val(relevance_time);
			}

			$('#elog-form-post').trigger( 'submit' );
		});
	
		$('#elog-post-reset').button().click(function() { that.post_reset(); });
	
		this.post_reset();
	};

	/* ------------------------------------------
	 *  Reset the post form to its initial state
	 * ------------------------------------------
	 */
	this.post_reset = function() {

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
			that.live_dim_all_highlights();
			that.live_message_viewer.refresh(live_selected_runs(), that.live_highlight);
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

	this.live_current_selected_range = '20';
	this.live_selected_range = function() {
		return this.live_current_selected_range;
	}

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
			'JSON').error(function () {
				alert('failed because of: '+jqXHR.statusText);
			});
		});
	};
	this.simple_search = function(text2search) {
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
		    author               = '';
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
			author
		);
	};
	this.search = function() {
		if(this.context2 == 'simple') {
			this.simple_search($('#elog-form-search input[name="text2search"]').val());
			return;
		}
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
			$('#elog-form-search select[name="author"]').val()
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

	this.expand_collapse = 0;
	this.days2threads = null;
	this.current_day = null;
	this.threads = null;
	this.messages = null;
	this.total_messages = 0;
	this.total_attachments = 0;
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
				$('#'+this.base+'-r-con-'+entry.id).html(
					'<div style="width:780px; height:300px; overflow:auto; padding:10px; background-color:#ffffff; ">'+data+'</div>'
				);
				entry.loaded = true;
			});
		} else {
			$(toggler).removeClass('ui-icon-triangle-1-s').addClass('ui-icon-triangle-1-e');
			$(container).removeClass('el-l-r-vis').addClass('el-l-r-hdn');
		}
	};

	this.expand_run = function(idx) {
    	var entry=this.threads[idx];
		var toggler='#'+this.base+'-r-tgl-'+entry.id;
		var container='#'+this.base+'-r-con-'+entry.id;
		if( $(container).hasClass('el-l-r-hdn')) {
			$(toggler).removeClass('ui-icon-triangle-1-e').addClass('ui-icon-triangle-1-s');
			$(container).removeClass('el-l-r-hdn').addClass('el-l-r-vis');
			$('#'+this.base+'-r-con-'+entry.id).html('Loading...');
			$.get('../logbook/DisplayRunParams.php',{id:entry.run_id},function(data) {
				var html = '<div style="width:820px; height:300px; overflow:auto; padding:10px; background-color:#ffffff; ">'+data+'</div>';
				$('#'+that.base+'-r-con-'+entry.id).html(html);
			});
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
	 * Check if new messages/rusn are aavailable, and if so - refresh the view.
	 * Highlight the new content if teh 'highlighter' function is passed as
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

					// Merge into ths object's data structures and extend DOM:
					//
					// - if no such day existed before then add the whole day
					// - otherwise merge this day's entries (messages & runs) into existing day
					// - expand messages according to the current state of other messages on the screen
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
									that.days2threads[day_idx].threads.push(thread_idx);
									var html = that.live_thread2html(thread_idx);
									$('#'+that.base+'-m-d-con-'+day_idx).prepend(html);
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
							that.days2threads.push(new_days2threads[i]);

							var day_idx = that.days2threads.length-1;
							var html = that.live_day2html(day_idx);
							$('#'+that.base+'-ms').prepend(html);

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

					$('.el-l-m-re').button();
					$('.el-l-m-ed').button();
					$('.el-l-m-mv').button();
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
		author) {

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
			author              : author
		};
		this.do_reload(params);
	};
	this.do_reload = function(params) {

		that.dim_day();

		$('#'+this.base+'-ms-info').html('Searching...');
		$.get('../logbook/Search.php',params,function(data) {

			var status = data.ResultSet.Status;
			if(status!='success') {
				$('#'+that.base+'-ms-info').html(data.ResultSet.Message);
				return;
			}
			$('#'+this.base+'-ms-info').html('Rendering...');

			that.expand_collapse = 0;
			that.threads = data.ResultSet.Result;
			that.messages = new Array();
			that.total_messages = 0;
			that.total_attachments = 0;
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

			var html = '';
			for(var day_idx = that.days2threads.length-1; day_idx >= 0; day_idx--)
				html += that.live_day2html(day_idx);
			$('#'+that.base+'-ms').html(html);

			if(that.days2threads.length) that.expand_group_day(that.days2threads.length-1, true);
			that.live_update_info();

			$('.el-l-m-re').button();
			$('.el-l-m-ed').button();
			$('.el-l-m-mv').button();

		},'json');
	};

	this.live_update_info = function() {
		$('#'+this.base+'-ms-info').html(
			'<center>Found: <b>'+that.total_messages+'</b> messages'+
			', <b>'+that.total_attachments+'</b> attachments'+
			(that.min_run ? ', runs: <b>'+that.min_run+'</b> .. <b>'+that.max_run+'</b>' : '')+
			'</center>'
		);
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
		var entry = this.messages[id];
		var html =
'<div id="'+this.base+'-m-rdlg-'+id+'" class="el-l-m-rdlg el-l-m-dlg-hdn">'+
'  <div style="float:left;">'+
'    <form id="elog-form-post-'+id+'" enctype="multipart/form-data" action="/apps-dev/logbook/NewFFEntry4portal.php" method="post">'+
'      <input type="hidden" name="id" value="'+this.parent.exp_id+'" />'+
'      <input type="hidden" name="scope" value="message" />'+
'      <input type="hidden" name="message_id" value="'+id+'" />'+
'      <input type="hidden" name="run_id" value="" />'+
'      <input type="hidden" name="shift_id" value="" />'+
'      <input type="hidden" name="MAX_FILE_SIZE" value="25000000" />'+
'      <input type="hidden" name="num_tags" value="0" />'+
'      <input type="hidden" name="onsuccess" value="" />'+
'      <input type="hidden" name="relevance_time" value="" />'+
'      <div style="font-weight:bold; margin-bottom:4px;">Reply:</div>'+
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
'  <div style="float:right;" class="s-b-con"><button class="el-l-m-re-c" id="'+this.base+'-m-re-c-'+id+'" onclick="'+this.address+'.live_message_reply_cancel('+id+');">cancel</button></div>'+
'  <div style="float:right;" class="s-b-con"><button class="el-l-m-re-s" id="'+this.base+'-m-re-s-'+id+'" onclick="'+this.address+'.live_message_reply_submit('+id+');">submit</button></div>'+
'  <div style="clear:both;"></div>'+
'</div>';
		if(this.parent.editor) {
			html +=
'<div id="'+this.base+'-m-edlg-'+id+'" class="el-l-m-edlg el-l-m-dlg-hdn">'+
'  <div style="float:left;">'+
'    <form id="elog-form-edit-'+id+'" enctype="multipart/form-data" action="/apps-dev/logbook/UpdateFFEntry4portal.php" method="post">'+
'      <input type="hidden" name="id" value="'+id+'" />'+
'      <input type="hidden" name="content_type" value="TEXT" />'+
'      <input type="hidden" name="onsuccess" value="" />'+
'      <div style="font-weight:bold; margin-bottom:4px;">Edit:</div>'+
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
'  <div style="float:right;" class="s-b-con"><button class="el-l-m-ed-c" id="'+this.base+'-m-ed-c-'+id+'" onclick="'+this.address+'.live_message_edit_cancel('+id+');">cancel</button></div>'+
'  <div style="float:right;" class="s-b-con"><button class="el-l-m-ed-s" id="'+this.base+'-m-ed-s-'+id+'" onclick="'+this.address+'.live_message_edit_submit('+id+');">submit</button></div>'+
'  <div style="clear:both;"></div>'+
'</div>'+
'<div id="'+this.base+'-m-mdlg-'+id+'" class="el-l-m-mdlg el-l-m-dlg-hdn">'+
'  <div style="float:right;" class="s-b-con"><button class="el-l-m-mv-c" id="'+this.base+'-m-mv-c-'+id+'" onclick="'+this.address+'.live_message_move_cancel('+id+');">cancel</button></div>'+
'  <div style="float:right;" class="s-b-con"><button class="el-l-m-mv-s" id="'+this.base+'-m-mv-s-'+id+'" onclick="'+this.address+'.live_message_move_submit('+id+');">submit</button></div>'+
'  <div style="clear:both;"></div>'+
''+
'  Here be the dialog allowing to move the message into some other context and/or time.'+
'  Right now we create it automatically for each message after loading it.'+
'  We should probably optimize things by creating it at first use.'+
'  That way we will reduce the size of DOM'+
'</div>';
		}

		dlgs.html(html);

		$('#'+this.base+'-m-re-s-'+id).button();
		$('#'+this.base+'-m-re-c-'+id).button();

		if(this.parent.editor) {
			$('#'+this.base+'-m-ed-s-'+id).button();
			$('#'+this.base+'-m-ed-c-'+id).button();
			$('#'+this.base+'-m-mv-e-'+id).button();
			$('#'+this.base+'-m-mv-c-'+id).button();
		}
	}

	this.live_enable_message_buttons = function(id, on) {
		var state = on ? 'enable' : 'disable';
		$('#'+this.base+'-m-re-'+id).button(state);
		$('#'+this.base+'-m-ed-'+id).button(state);
		$('#'+this.base+'-m-mv-'+id).button(state);
	}

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
		$('#elog-form-post-'+id+' input[name="onsuccess"]').val(urlbase+'&page1=elog');

		$('#elog-form-post-'+id).trigger( 'submit' );

	};
	this.live_message_edit_submit = function(id) {
		var urlbase = window.location.href;
		var idx = urlbase.indexOf( '&' );
		if( idx > 0 ) urlbase = urlbase.substring( 0, idx );
		$('#elog-form-edit-'+id+' input[name="onsuccess"]').val(urlbase+'&page1=elog');

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
'    <div style="float:left; margin-left:10px;" class="el-l-m-subj">'+entry.subject+'</div>'+message_signs(entry)+
'    <div style="clear:both;"></div>'+
'  </div>'+
'  <div class="el-l-m-con el-l-m-hdn" id="'+this.base+'-m-con-'+entry.id+'">'+
'    <div class="el-l-m-body">';
			if(this.parent.editor)
				html +=
'      <div style="float:right;" class="s-b-con"><button class="el-l-m-mv"  id="'+this.base+'-m-mv-'+entry.id+'"  onclick="'+this.address+'.live_message_move('+entry.id+',false);">move</button></div>'+
'      <div style="float:right;" class="s-b-con"><button class="el-l-m-ed"  id="'+this.base+'-m-ed-'+entry.id+'"  onclick="'+this.address+'.live_message_edit('+entry.id+',false);">edit</button></div>';
			html +=
'      <div style="float:right;" class="s-b-con"><button class="el-l-m-re" id="'+this.base+'-m-re-'+entry.id+'" onclick="'+this.address+'.live_message_reply('+entry.id+',false);">reply</button></div>'+
'      <div style="float:left; font-size:12px;">'+entry.html1+'</div>'+
'      <div style="clear:both;"></div>'+
'      <div id="'+this.base+'-m-dlgs-'+entry.id+'"></div>'+
'    </div>';
			that.attachments[entry.id] = entry.attachments;
			var attachments_html = '';
			for(var k in entry.attachments) {
				that.total_attachments++;
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
//'          <a href="../logbook/attachments/'+a.id+'/'+a.description+'" target="_blank"><img src="../logbook/attachments/preview/'+a.id+'" /></a>'+
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
			for(var k in entry.children) html += that.live_child2html(eval("("+entry.children[k]+")"));
				html +=
'  </div>';
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
		}
		return html;
	};

	this.live_child2html = function(entry) {
		that.messages[entry.id] = entry;
		var html =
'<div class="el-l-c-hdr" onclick="'+this.address+'.toggle_child('+entry.id+');">'+
'  <div style="float:left;"><span class="toggler ui-icon ui-icon-triangle-1-e el-l-c-tgl" id="'+this.base+'-c-tgl-'+entry.id+'"></span></div>'+
'  <div style="float:left;" class="el-l-c-time">'+entry.relevance_time+'</div>'+
'  <div style="float:left;" class="el-l-c-author">'+entry.author+'</div>'+
'  <div style="float:left; margin-left:10px;" class="el-l-c-subj">'+entry.subject+'</div>'+message_signs(entry)+
'  <div style="clear:both;"></div>'+
'</div>'+
'<div class="el-l-c-con el-l-c-hdn" id="'+this.base+'-c-con-'+entry.id+'">'+
'  <div class="el-l-c-body">'+
'    <div style="float:left; font-size:12px;">'+entry.html1+'</div>';
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
			that.total_attachments++;
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

		var children = entry.children;
		for(var i in children) html += that.live_child2html(eval("("+children[i]+")"));

		html +=
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

	/* ATTENTION: Watch for dependencies! These functiona will call a function from
	 *            the application context.
	 *
	 * TODO: Re-evaluate this code to see if the dependencies are properly
	 *       designed and used.
	 */
	this.highlight_day = function(idx) {
		if(that.current_day == idx) return;
		that.current_day = idx;
		var day = that.days2threads[idx];
		applications['p-appl-elog'].context3 = '<a class="link" href="#'+this.base+'-m-d-hdr-'+idx+'" title="go to the day header">'+day.ymd+'</a>';
		set_context(applications['p-appl-elog']);
	};

	this.dim_day = function() {
		if(that.current_day == null) return;
		that.current_day = null;
		applications['p-appl-elog'].context3 ='';
		set_context(applications['p-appl-elog']);
	};
}
