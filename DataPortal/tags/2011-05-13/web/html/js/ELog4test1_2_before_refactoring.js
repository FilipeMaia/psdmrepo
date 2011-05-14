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
	this.select    = function(ctx1, ctx2) {
		that.context1 = ctx1;
		that.context2 = ctx2 == null ? context2_default[ctx1] : ctx2;
		that.context3 ='';
		if(that.context1 == 'recent') {

			elog.live_current_selected_range = that.context2;
			elog.live_reload();

		} else if(that.context1 == 'post') {

			$('#elog-form-post input:[name="scope"]').val(that.context1);
			if(that.context2 == 'experiment') {

				$('#el-p-message4experiment').removeClass('hidden' ).addClass('visible');
				$('#el-p-message4shift'     ).removeClass('visible').addClass('hidden');
				$('#el-p-message4run'       ).removeClass('visible').addClass('hidden');

			} else if(that.context2 == 'shift') {

				$('#el-p-message4experiment').removeClass('visible').addClass('hidden');
				$('#el-p-message4shift'     ).removeClass('hidden' ).addClass('visible');
				$('#el-p-message4run'       ).removeClass('visible').addClass('hidden');

			} else if(that.context2 == 'run') {

				$('#el-p-message4experiment').removeClass('visible').addClass('hidden');
				$('#el-p-message4shift'     ).removeClass('visible').addClass('hidden');
				$('#el-p-message4run'       ).removeClass('hidden' ).addClass('visible');
			}
			that.post_reset();
		}
	};

	/*
	 * NOTE: For the moment this is done statically.
	 * 
	 * TODO: Consider dynamic initialization of the elements shown below.
	 *
	 *   $('#p-center > #application-workarea > #elog').html('<center>The workarea of the e-log</center>');
	 *   $('#p-left > #v-menu > #elog').html('<center>The menu area of the e-log</center>');
     */

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

	/* --------------------------------------------
	 *  Initialize the form for searching messages
	 * --------------------------------------------
	 */
	this.search_init = function() {
		$('#elog-search-submit').button().click(function() {
		});
		$('#elog-search-reset').button().click(function() { that.search_reset(); });
	};

	/* --------------------------------------------
	 *  Reset the search form to its initial state
	 * --------------------------------------------
	 */
	this.search_reset = function() {
	};

	/* -------------------------------------------
	 *  Initialize the 'Live' message display tab
	 * -------------------------------------------
	 */
	this.live_expand_collapse = 0;
	this.live_days2threads = null;
	this.live_current_day = null;
	this.live_threads = null;
	this.live_messages = null;
	this.live_total_messages = 0;
	this.live_runs = null;
	this.live_min_run = 0;
	this.live_max_run = 0;
	this.live_attachments = null;
	this.live_attachments_loader = null;

	this.live_init = function() {

		$('#el-l-rs-selector').buttonset().change(function() {
			elog.live_dim_day();
			elog.live_reload();
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
		$('#el-l-expand').button().click(function() {
			switch( that.live_expand_collapse ) {
			default:
			case 1:
				that.live_expand_all_messages();
			case 0:
				$('.el-l-m-d-tgl').removeClass('ui-icon-triangle-1-e').addClass('ui-icon-triangle-1-s');
				$('.el-l-m-d-con').removeClass('el-l-m-d-hdn').addClass('el-l-m-d-vis');
			}
			that.live_expand_collapse++;
			if( that.live_expand_collapse > 2 ) that.live_expand_collapse = 2;
		});
		$('#el-l-collapse').button().click(function() {
			that.live_dim_day();
			switch( that.live_expand_collapse ) {
			default:
			case 1:
				$('.el-l-m-d-tgl').removeClass('ui-icon-triangle-1-s').addClass('ui-icon-triangle-1-e');
				$('.el-l-m-d-con').removeClass('el-l-m-d-vis').addClass('el-l-m-d-hdn');
			case 2:
				$('.el-l-m-tgl').removeClass('ui-icon-triangle-1-s').addClass('ui-icon-triangle-1-e');
				$('.el-l-m-con').removeClass('el-l-m-vis').addClass('el-l-m-hdn');
				that.live_collapse_all_runs();
			}
			that.live_expand_collapse--;
			if( that.live_expand_collapse < 0 ) that.live_expand_collapse = 0;
		});
		$('#el-l-viewattach').button().click(function() {
			that.live_dim_day();
			$('.el-l-a-tgl').removeClass('ui-icon-triangle-1-e').addClass('ui-icon-triangle-1-s');
			$('.el-l-a-con').removeClass('el-l-a-hdn').addClass('el-l-a-vis');
		});
		$('#el-l-hideattach').button().click(function() {
			that.live_dim_day();
			$('.el-l-a-tgl').removeClass('ui-icon-triangle-1-s').addClass('ui-icon-triangle-1-e');
			$('.el-l-a-con').removeClass('el-l-a-vis').addClass('el-l-a-hdn');
		});
		$('#el-l-refresh').button().click(function() {
			that.live_dim_all_highlights();
			that.live_refresh();
		});
		$('#el-l-refresh-interval').change(function(ev) {
			that.live_stop_refresh();
			that.live_schedule_refresh();
		});

		this.live_reload();
		this.live_schedule_refresh();
		this.live_start_highlight_timer();
	};

	this.live_expand_group_day = function(idx, on) {
		var toggler='#el-l-m-d-tgl-'+idx;
		var container='#el-l-m-d-con-'+idx;
		if(on) {
			that.live_highlight_day(idx);
			$(toggler).removeClass('ui-icon-triangle-1-e').addClass('ui-icon-triangle-1-s');
			$(container).removeClass('el-l-m-d-hdn').addClass('el-l-m-d-vis');
		} else {
			that.live_dim_day();
			$(toggler).removeClass('ui-icon-triangle-1-s').addClass('ui-icon-triangle-1-e');
			$(container).removeClass('el-l-m-d-vis').addClass('el-l-m-d-hdn');
		}
	};

	this.live_toggle_group_day = function(idx) {
		var container='#el-l-m-d-con-'+idx;
		that.live_expand_group_day(idx, $(container).hasClass('el-l-m-d-hdn'));
	};

	this.live_expand_message = function(idx, on) {
		var entry = that.live_threads[idx];
		var toggler='#el-l-m-tgl-'+entry.id;
		var container='#el-l-m-con-'+entry.id;
		if(on) {
			$(toggler).removeClass('ui-icon-triangle-1-e').addClass('ui-icon-triangle-1-s');
			$(container).removeClass('el-l-m-hdn').addClass('el-l-m-vis');
			var attachments = entry.attachments;
			for(var i = 0; i < attachments.length; i++) {
				this.live_expand_attachment(attachments[i].id, true);
			}
			var children = entry.children;
			for(var i = 0; i < children.length; i++) {
				var child = eval( "("+children[i]+")" );
				this.live_expand_child(child.id, true);
			}
		} else {
			$(toggler).removeClass('ui-icon-triangle-1-s').addClass('ui-icon-triangle-1-e');
			$(container).removeClass('el-l-m-vis').addClass('el-l-m-hdn');
		}
	};

	this.live_expand_all_messages = function() {
		for(var i=0; i < that.live_threads.length; i++)
			this.live_expand_message(i, true);
	};

	this.live_toggle_message = function(idx) {
		var entry = that.live_threads[idx];
		var container='#el-l-m-con-'+entry.id;
		this.live_expand_message(idx, $(container).hasClass('el-l-m-hdn'));
	};

	function lengthof(a) {
		var result=0;
		for(var i in a) result++;
		return result;
	}

	this.live_expand_child = function(id, on) {
		var entry = that.live_messages[id];
		var toggler='#el-l-c-tgl-'+id;
		var container='#el-l-c-con-'+id;
		if(on) {
			$(toggler).removeClass('ui-icon-triangle-1-e').addClass('ui-icon-triangle-1-s');
			$(container).removeClass('el-l-c-hdn').addClass('el-l-c-vis');
			var attachments = entry.attachments;
			for(var i = 0; i < attachments.length; i++) {
				this.live_expand_attachment(attachments[i].id, true);
			}
			var children = entry.children;
			for(var i = 0; i < children.length; i++) {
				var child = eval( "("+children[i]+")" );
				this.live_expand_child(child.id, true);
			}
		} else {
			$(toggler).removeClass('ui-icon-triangle-1-s').addClass('ui-icon-triangle-1-e');
			$(container).removeClass('el-l-c-vis').addClass('el-l-c-hdn');
		}
	};

	this.live_toggle_child = function(id) {
		var container='#el-l-c-con-'+id;
		this.live_expand_child(id, $(container).hasClass('el-l-c-hdn'));
	};

	this.live_expand_attachment = function(id, on) {
		var toggler='#el-l-a-tgl-'+id;
		var container='#el-l-a-con-'+id;
		if(on) {
			var a = that.live_attachments_loader[id];
			if(!a.loaded) {
				a.loaded = true;
			    var t = a.descr.type.split('/');
			    var html = '';
//			    if(( t[0] == 'text' ) || ( t[0] == 'application' && ( t[1] == 'octet-stream' || t[1] == 'x-octet-stream' ))) {
//			    	html = '<div style="max-width:800px; min-height:40px; max-height:200px; overflow:auto; border:solid 1px;"><textbox><pre>To be implemented</pre></textbox></div>';
//		        } else {
			    	html = '<a href="../logbook/attachments/'+id+'/'+a.descr.description+'" target="_blank"><img src="../logbook/attachments/preview/'+id+'" /></a>';
//		        }
				$(container).html(html);
			}
			$(toggler).removeClass('ui-icon-triangle-1-e').addClass('ui-icon-triangle-1-s');
			$(container).removeClass('el-l-a-hdn').addClass('el-l-a-vis');
		} else {
			$(toggler).removeClass('ui-icon-triangle-1-s').addClass('ui-icon-triangle-1-e');
			$(container).removeClass('el-l-a-vis').addClass('el-l-a-hdn');
		}
	}

	this.live_toggle_attachment = function(id) {
		var container='#el-l-a-con-'+id;
		this.live_expand_attachment(id, $(container).hasClass('el-l-a-hdn'));
	};

	this.live_toggle_run = function(idx) {
		var entry=this.live_threads[idx];
		var toggler='#el-l-r-tgl-'+entry.id;
		var container='#el-l-r-con-'+entry.id;
		if( $(container).hasClass('el-l-r-hdn')) {
			$(toggler).removeClass('ui-icon-triangle-1-e').addClass('ui-icon-triangle-1-s');
			$(container).removeClass('el-l-r-hdn').addClass('el-l-r-vis');
			if(entry.loaded) return;
			$('#el-l-r-con-'+entry.id).html('Loading...');
			$.get('../logbook/DisplayRunParams.php',{id:entry.run_id},function(data) {
				$('#el-l-r-con-'+entry.id).html(
					'<div style="width:780px; height:300px; overflow:auto; padding:10px; background-color:#ffffff; ">'+data+'</div>'
				);
				entry.loaded = true;
			});
		} else {
			$(toggler).removeClass('ui-icon-triangle-1-s').addClass('ui-icon-triangle-1-e');
			$(container).removeClass('el-l-r-vis').addClass('el-l-r-hdn');
		}
	};

	this.live_expand_run = function(idx) {
    	var entry=this.live_threads[idx];
		var toggler='#el-l-r-tgl-'+entry.id;
		var container='#el-l-r-con-'+entry.id;
		if( $(container).hasClass('el-l-r-hdn')) {
			$(toggler).removeClass('ui-icon-triangle-1-e').addClass('ui-icon-triangle-1-s');
			$(container).removeClass('el-l-r-hdn').addClass('el-l-r-vis');
			$('#el-l-r-con-'+entry.id).html('Loading...');
			$.get('../logbook/DisplayRunParams.php',{id:entry.run_id},function(data) {
				var html = '<div style="width:820px; height:300px; overflow:auto; padding:10px; background-color:#ffffff; ">'+data+'</div>';
				$('#el-l-r-con-'+entry.id).html(
					html
				);
			});
		}
	};

	this.live_collapse_run = function(idx) {
	   	var entry=this.live_threads[idx];
		var toggler='#el-l-r-tgl-'+entry.id;
		var container='#el-l-r-con-'+entry.id;
		if( $(container).hasClass('el-l-r-vis')) {
			$(toggler).removeClass('ui-icon-triangle-1-s').addClass('ui-icon-triangle-1-e');
			$(container).removeClass('el-l-r-vis').addClass('el-l-r-hdn');
		}
	};

	this.live_collapse_all_runs = function() {
		for(var i in this.live_runs) {
			this.live_collapse_run(this.live_runs[i]);
		}
	};

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
	    that.live_highlight_timer = window.setTimeout('elog.live_highlight_actions()',live_refresh_interval());
	};
	this.live_highlight_actions = function() {
		for(var id in that.live_highlighted) {
			that.live_highlighted[id] -= live_refresh_interval();
			if(that.live_highlighted[id] <= 0) {
				$(id).removeClass('el-l-m-highlight');
				delete that.live_highlighted[id];
			}
		}
		that.live_start_highlight_timer();
	};
	this.live_dim_all_highlights = function() {
		for(var id in that.live_highlighted) {
			that.live_highlighted[id] -= live_refresh_interval();
			$(id).removeClass('el-l-m-highlight');
			delete that.live_highlighted[id];
		}
	};
	this.live_stop_highlight_timer = function() {
		that.live_dim_all_highlights();
	    if(that.live_highlight_timer != null) {
	        window.clearTimeout(that.live_highlight_timer);
	        that.live_highlight_timer = null;
	    }
	};

	this.live_refresh_timer = null;
	this.live_schedule_refresh = function() {
	    that.live_refresh_timer = window.setTimeout('elog.live_refresh_actions()',live_refresh_interval());
	};
	this.live_stop_refresh = function() {
	    if(that.live_refresh_timer != null) {
	        window.clearTimeout(that.live_refresh_timer);
	        that.live_refresh_timer = null;
	    }
	};
	this.live_refresh_actions = function() {
		that.live_refresh();
		that.live_schedule_refresh();
	};
	this.live_refresh = function() {
		if(that.live_threads.length) {

			//$('#el-l-ms-action').html('<image src="../logbook/images/ajaxloader.gif" />');

			var params = {
				id: this.exp_id,
				scope: 'experiment',
				search_in_messages: 1,
				search_in_tags: 1,
				search_in_values: 1,
				posted_at_experiment: 1,
				posted_at_shift: 1,
				posted_at_run: 1,
				format: 'detailed',
				since: that.live_threads[that.live_threads.length-1].event_timestamp
			};
			if( live_selected_runs()) params.inject_runs = '';

			$.get('../logbook/Search.php',params,function(data) {

				var status = data.ResultSet.Status;
				if(status!='success') {
					$('#el-l-ms-action').html('status: '+status+', message: '+data.ResultSet.Message);
					return;
				}
				$('#el-l-ms-action').html('');

				var threads = data.ResultSet.Result;
				if(threads.length) {

					var days2threads = new Array();

					var last_day = undefined;
					for(var i=0; i < threads.length; i++) {
						var entry = threads[i];
						var ymd = entry.ymd;
						if((days2threads.length == 0) || (last_day != ymd)) {
							days2threads.push( { ymd:ymd, runs:0, min_run:0, max_run:0, messages:0, threads: new Array() } );
							last_day = ymd;
						}
						var idx = days2threads.length - 1;
						if(entry.is_run) {
							entry.run_num = parseInt(entry.run_num);
							days2threads[idx].runs++;
							if((days2threads[idx].min_run == 0) || (entry.run_num < days2threads[idx].min_run)) days2threads[idx].min_run = entry.run_num;
							if((days2threads[idx].max_run == 0) || (entry.run_num > days2threads[idx].max_run)) days2threads[idx].max_run = entry.run_num;
						} else {
							days2threads[idx].messages++;
						}
						that.live_threads.push(entry);
						var thread_idx = that.live_threads.length-1;
						days2threads[idx].threads.push(thread_idx);
					}

					// Merge into ths object's data structures and extend DOM:
					//
					// - if no such day existed before then add the whole day
					// - otherwise merge this day's entries (messages & runs) into existing day
					// - expand messages according to the current state of other messages on the screen
					//
					for(var i = days2threads.length-1; i >= 0; i--) {
						var found = false;
						for(var day_idx = that.live_days2threads.length-1; day_idx >= 0; day_idx--) {

							var day = that.live_days2threads[day_idx];
							if(that.live_days2threads[day_idx].ymd == days2threads[i].ymd) {

								that.live_days2threads[day_idx].messages += days2threads[i].messages;
								that.live_days2threads[day_idx].runs += days2threads[i].runs;

								if(days2threads[i].min_run && (days2threads[i].min_run < that.live_days2threads[day_idx].min_run)) that.live_days2threads[day_idx].min_run = days2threads[i].min_run;
								if(days2threads[i].max_run && (days2threads[i].max_run > that.live_days2threads[day_idx].max_run)) that.live_days2threads[day_idx].max_run = days2threads[i].max_run;

								for(var j = 0; j < days2threads[i].threads.length; j++) {
									var thread_idx = days2threads[i].threads[j];
									that.live_days2threads[day_idx].threads.push(thread_idx);
									var html = that.live_thread2html(thread_idx);
									$('#el-l-m-d-con-'+day_idx).prepend(html);
								}
								var runs_messages_info = '<b>'+day.messages+'</b> messages'+( day.runs ? ', runs: <b>'+day.min_run+'</b> .. <b>'+day.max_run+'</b>' : '' );
								$('#el-l-m-d-con-info-'+day_idx).html(runs_messages_info);

								that.live_expand_group_day(day_idx, true);
								that.live_highlight('#el-l-m-d-hdr-'+day_idx);

								found = true;
								break;
							}
						}
						if(!found) {
							that.live_days2threads.push(days2threads[i]);

							var day_idx = that.live_days2threads.length-1;
							var html = that.live_day2html(day_idx);
							$('#el-l-ms').prepend(html);

							that.live_expand_group_day(day_idx, true);
							that.live_highlight('#el-l-m-d-hdr-'+day_idx);
						}
						for(var j = 0; j < days2threads[i].threads.length; j++) {
							var thread_idx = days2threads[i].threads[j];
							if(that.live_expand_collapse > 1) that.live_expand_message(thread_idx, true);
							that.live_highlight('#el-l-m-hdr-'+thread_idx);
						}
					}
					that.live_update_info();

					$('.el-l-m-re').button();
					$('.el-l-m-ed').button();
					$('.el-l-m-mv').button();
				}

			},'json');

		} else {
			that.live_reload();
		}
	};

	this.live_reload = function() {

		that.live_dim_day();
		that.live_dim_all_highlights();

		$('#el-l-ms-action').html('Searching...');

		var params = {
			id: this.exp_id,
			scope: 'experiment',
			search_in_messages: 1,
			search_in_tags: 1,
			search_in_values: 1,
			posted_at_experiment: 1,
			posted_at_shift: 1,
			posted_at_run: 1,
			format: 'detailed'
		};
		if( live_selected_runs()) params.inject_runs = '';

		var limit = that.live_selected_range();
		if(limit) params.limit = limit;

		$.get('../logbook/Search.php',params,function(data) {

			var status = data.ResultSet.Status;
			if(status!='success') {
				$('#el-l-ms-action').html('status: '+status+', message: '+data.ResultSet.Message);
				return;
			}
			$('#el-l-ms-action').html('');

			that.live_expand_collapse = 0;
			that.live_threads = data.ResultSet.Result;
			that.live_messages = new Array();
			that.live_total_messages = 0;
			that.live_runs = new Array();
			that.live_min_run = 0;
			that.live_max_run = 0;
			that.live_attachments = new Array();
			that.live_attachments_loader = new Array();
			that.live_days2threads = new Array();

			var last_day = undefined;
			for(var i=0; i < that.live_threads.length; i++) {
				var entry = that.live_threads[i];
				var ymd = entry.ymd;
				if((that.live_days2threads.length == 0) || (last_day != ymd)) {
					that.live_days2threads.push( { ymd:ymd, runs:0, min_run:0, max_run:0, messages:0, threads: new Array() } );
					last_day = ymd;
				}
				var idx = that.live_days2threads.length - 1;
				if( entry.is_run ) {
					entry.run_num = parseInt(entry.run_num);
					that.live_days2threads[idx].runs++;
					if((that.live_days2threads[idx].min_run == 0) || (entry.run_num < that.live_days2threads[idx].min_run)) that.live_days2threads[idx].min_run = entry.run_num;
					if((that.live_days2threads[idx].max_run == 0) || (entry.run_num > that.live_days2threads[idx].max_run)) that.live_days2threads[idx].max_run = entry.run_num;
				} else {
					that.live_days2threads[idx].messages++;
				}
				that.live_days2threads[idx].threads.push(i);
			}

			var html = '';
			for(var day_idx = that.live_days2threads.length-1; day_idx >= 0; day_idx--)
				html += that.live_day2html(day_idx);
			$('#el-l-ms').html(html);

			that.live_expand_group_day(that.live_days2threads.length-1, true);
			that.live_update_info();

			$('.el-l-m-re').button();
			$('.el-l-m-ed').button();
			$('.el-l-m-mv').button();

		},'json');
	};

	this.live_update_info = function() {
		var n_attachments = 0;
		for(var a in that.live_attachments) n_attachments++;
		$('#el-l-ms-info').html(
			'<center>Found: <b>'+that.live_total_messages+'</b> messages'+
			', <b>'+n_attachments+'</b> attachments'+
			(that.live_min_run ? ', runs: <b>'+that.live_min_run+'</b> .. <b>'+that.live_max_run+'</b>' : '')+
			'</center>'
		);
	};
	this.live_day2html = function(day_idx) {
		var html = '';
		var day = that.live_days2threads[day_idx];
		var runs_messages_info = '<b>'+day.messages+'</b> messages'+( day.runs ? ', runs: <b>'+day.min_run+'</b> .. <b>'+day.max_run+'</b>' : '' );
		html +=
'<div class="el-l-m-d-hdr" id="el-l-m-d-hdr-'+day_idx+'" onclick="elog.live_toggle_group_day('+day_idx+');">'+
'  <div style="float:left;">'+
'    <span class="ui-icon ui-icon-triangle-1-e el-l-m-d-tgl" id="el-l-m-d-tgl-'+day_idx+'"></span>'+
'  </div>'+
'  <div style="float:left;" class="el-l-m-d">'+day.ymd+'</div>'+
'  <div style="float:right; margin-right:10px;" id="el-l-m-d-con-info-'+day_idx+'">'+runs_messages_info+'</div>'+
'  <div style="clear:both;"></div>'+
'</div>'+
'<div class="el-l-m-d-con el-l-m-d-hdn" id="el-l-m-d-con-'+day_idx+'" onmouseover="elog.live_highlight_day('+"'"+day_idx+"'"+');">';

		for(var i = day.threads.length-1; i >= 0; i--) {
			var thread_idx = day.threads[i];
			html += that.live_thread2html(thread_idx);
   		}
			html +=
'</div>';
		return html;
	};

	this.create_live_message_dialogs = function(id, is_child) {
		var dlgs = $('#el-l-m-dlgs-'+id);
		if(dlgs.html() != '') return;
		var entry = that.live_messages[id];
		var html =
'<div id="el-l-m-rdlg-'+id+'" class="el-l-m-rdlg el-l-m-dlg-hdn">'+
'  <div style="float:left;">'+
'    <form id="elog-form-post-'+id+'" enctype="multipart/form-data" action="/apps-dev/logbook/NewFFEntry4portal.php" method="post">'+
'      <input type="hidden" name="id" value="'+that.exp_id+'" />'+
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
'          <input type="text" name="author_account" value="'+that.author+'" size=32 style="padding:2px; margin-top:5px;" />'+
'        </div>'+
'        <div style="float:left; margin-left:20px;">'+ 
'          <div style="font-weight:bold;">Attachments:</div>'+
'          <div id="el-l-reply-as-'+id+'" style="margin-top:5px;">'+
'            <div>'+
'              <input type="file" name="file2attach_0" onchange="elog.live_reply_add_attachment('+id+')" />'+
'              <input type="hidden" name="file2attach_0" value=""/ >'+
'            </div>'+
'          </div>'+
'        </div>'+
'        <div style="clear:both;"></div>'+
'      </div>'+
'    </form>'+
'  </div>'+
'  <div style="float:right;" class="s-b-con"><button class="el-l-m-re-c" id="el-l-m-re-c-'+id+'" onclick="elog.live_message_reply_cancel('+id+');">cancel</button></div>'+
'  <div style="float:right;" class="s-b-con"><button class="el-l-m-re-s" id="el-l-m-re-s-'+id+'" onclick="elog.live_message_reply_submit('+id+');">submit</button></div>'+
'  <div style="clear:both;"></div>'+
'</div>';
		if(that.editor) {
			html +=
'<div id="el-l-m-edlg-'+id+'" class="el-l-m-edlg el-l-m-dlg-hdn">'+
'  <div style="float:left;">'+
'    <form id="elog-form-edit-'+id+'" enctype="multipart/form-data" action="/apps-dev/logbook/UpdateFFEntry4portal.php" method="post">'+
'      <input type="hidden" name="id" value="'+id+'" />'+
'      <input type="hidden" name="content_type" value="TEXT" />'+
'      <input type="hidden" name="onsuccess" value="" />'+
'      <div style="font-weight:bold; margin-bottom:4px;">Edit:</div>'+
'      <textarea name="content" rows="12" cols="64" style="padding:4px;" title="the first line of the message body will be used as its subject" >'+entry.content+'</textarea>'+
'      <div style="font-weight:bold; margin-top: 10px;">Extra attachments:</div>'+
'      <div id="el-l-edit-as-'+id+'" style="margin-top:5px;">'+
'        <div>'+
'          <input type="file" name="file2attach_0" onchange="elog.live_edit_add_attachment('+id+')" />'+
'          <input type="hidden" name="file2attach_0" value=""/ >'+
'        </div>'+
'      </div>'+
'    </form>'+
'  </div>'+
'  <div style="float:right;" class="s-b-con"><button class="el-l-m-ed-c" id="el-l-m-ed-c-'+id+'" onclick="elog.live_message_edit_cancel('+id+');">cancel</button></div>'+
'  <div style="float:right;" class="s-b-con"><button class="el-l-m-ed-s" id="el-l-m-ed-s-'+id+'" onclick="elog.live_message_edit_submit('+id+');">submit</button></div>'+
'  <div style="clear:both;"></div>'+
'</div>'+
'<div id="el-l-m-mdlg-'+id+'" class="el-l-m-mdlg el-l-m-dlg-hdn">'+
'  <div style="float:right;" class="s-b-con"><button class="el-l-m-mv-c" id="el-l-m-mv-c-'+id+'" onclick="elog.live_message_move_cancel('+id+');">cancel</button></div>'+
'  <div style="float:right;" class="s-b-con"><button class="el-l-m-mv-s" id="el-l-m-mv-s-'+id+'" onclick="elog.live_message_move_submit('+id+');">submit</button></div>'+
'  <div style="clear:both;"></div>'+
''+
'  Here be the dialog allowing to move the message into some other context and/or time.'+
'  Right now we create it automatically for each message after loading it.'+
'  We should probably optimize things by creating it at first use.'+
'  That way we will reduce the size of DOM'+
'</div>';
		}

		dlgs.html(html);

		$('#el-l-m-re-s-'+id).button();
		$('#el-l-m-re-c-'+id).button();

		if(that.editor) {
			$('#el-l-m-ed-s-'+id).button();
			$('#el-l-m-ed-c-'+id).button();
			$('#el-l-m-mv-e-'+id).button();
			$('#el-l-m-mv-c-'+id).button();
		}
	}

	this.live_enable_message_buttons = function(id, on) {
		var state = on ? 'enable' : 'disable';
		$('#el-l-m-re-'+id).button(state);
		$('#el-l-m-ed-'+id).button(state);
		$('#el-l-m-mv-'+id).button(state);
	}

	this.live_message_reply = function(id, is_child) {
	    that.create_live_message_dialogs(id, is_child);
		$('#el-l-m-rdlg-'+id).removeClass('el-l-m-dlg-hdn').addClass('el-l-m-dlg-vis');
		that.live_enable_message_buttons(id, false);
		$('#el-l-m-rdlg-'+id+' textarea').focus();
	};
	this.live_message_edit = function(id, is_child) {
	    that.create_live_message_dialogs(id, is_child);
		$('#el-l-m-edlg-'+id).removeClass('el-l-m-dlg-hdn').addClass('el-l-m-dlg-vis');
		that.live_enable_message_buttons(id, false);
		$('#el-l-m-edlg-'+id+' textarea').focus();
	};
	this.live_message_move = function(id, is_child) {
	    that.create_live_message_dialogs(id, is_child);
		$('#el-l-m-mdlg-'+id).removeClass('el-l-m-dlg-hdn').addClass('el-l-m-dlg-vis');
		that.live_enable_message_buttons(id, false);
	};

	this.live_message_reply_cancel = function(id) {
		$('#el-l-m-rdlg-'+id).removeClass('el-l-m-dlg-vis').addClass('el-l-m-dlg-hdn');
		that.live_enable_message_buttons(id, true);
	};
	this.live_message_edit_cancel = function(id) {
		$('#el-l-m-edlg-'+id).removeClass('el-l-m-dlg-vis').addClass('el-l-m-dlg-hdn');
		that.live_enable_message_buttons(id, true);
	};
	this.live_message_move_cancel = function(id) {
		$('#el-l-m-mdlg-'+id).removeClass('el-l-m-dlg-vis').addClass('el-l-m-dlg-hdn');
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
		$('#el-l-m-mdlg-'+id).removeClass('el-l-m-dlg-vis').addClass('el-l-m-dlg-hdn');
		that.live_enable_message_buttons(id, true);
	};

	this.live_reply_add_attachment = function(id) {
		var num = $('#el-l-reply-as-'+id+' > div').size();
		$('#el-l-reply-as-'+id).append(
'  <div>'+
'    <input type="file" name="file2attach_'+num+'" onchange="elog.live_reply_add_attachment('+id+')" />'+
'    <input type="hidden" name="file2attach_'+num+'" value=""/ >'+
'  </div>'
		);
	};
	this.live_edit_add_attachment = function(id) {
		var num = $('#el-l-edit-as-'+id+' > div').size();
		$('#el-l-edit-as-'+id).append(
'  <div>'+
'    <input type="file" name="file2attach_'+num+'" onchange="elog.live_edit_add_attachment('+id+')" />'+
'    <input type="hidden" name="file2attach_'+num+'" value=""/ >'+
'  </div>'
		);
	};

	this.live_thread2html = function(thread_idx) {
		var html = '';
		var entry = that.live_threads[thread_idx];
		if(!entry.is_run) {
			that.live_messages[entry.id] = entry;
			that.live_total_messages++;
			html +=
'  <div class="el-l-m-hdr" id="el-l-m-hdr-'+thread_idx+'" onclick="elog.live_toggle_message('+thread_idx+');">'+
'    <div style="float:left;"><span class="ui-icon ui-icon-triangle-1-e el-l-m-tgl" id="el-l-m-tgl-'+entry.id+'"></span></div>'+
'    <div style="float:left;" class="el-l-m-time">'+entry.hms+'</div>'+
'    <div style="float:left;" class="el-l-m-author">'+entry.author+'</div>'+
'    <div style="float:left; margin-left:10px;" class="el-l-m-subj">'+entry.subject+'</div>'+message_signs(entry)+
'    <div style="clear:both;"></div>'+
'  </div>'+
'  <div class="el-l-m-con el-l-m-hdn" id="el-l-m-con-'+entry.id+'">'+
'    <div class="el-l-m-body">';
			if(that.editor)
				html +=
'      <div style="float:right;" class="s-b-con"><button class="el-l-m-mv"  id="el-l-m-mv-'+entry.id+'"  onclick="elog.live_message_move('+entry.id+',false);">move</button></div>'+
'      <div style="float:right;" class="s-b-con"><button class="el-l-m-ed"  id="el-l-m-ed-'+entry.id+'"  onclick="elog.live_message_edit('+entry.id+',false);">edit</button></div>';
			html +=
'      <div style="float:right;" class="s-b-con"><button class="el-l-m-re" id="el-l-m-re-'+entry.id+'" onclick="elog.live_message_reply('+entry.id+',false);">reply</button></div>'+
'      <div style="float:left; font-size:12px;">'+entry.html1+'</div>'+
'      <div style="clear:both;"></div>'+
'      <div id="el-l-m-dlgs-'+entry.id+'"></div>'+
'    </div>';
			that.live_attachments[entry.id] = entry.attachments;
			var attachments_html = '';
			for(var k in entry.attachments) {
				var a = entry.attachments[k];
				that.live_attachments_loader[a.id] = {loaded: false, descr: a};
				attachments_html +=
'      <div style="float:left;" class="el-l-a">'+
'        <div style="float:left;">'+
'          <span class="ui-icon ui-icon-triangle-1-e el-l-a-tgl" id="el-l-a-tgl-'+a.id+'" onclick="elog.live_toggle_attachment('+a.id+');"></span>'+
'        </div>'+
'        <div style="float:left;" class="el-l-a-dsc"><a class="link" href="../logbook/attachments/'+a.id+'/'+a.description+'"  target="_blank">'+a.description+'</a></div>'+
'        <div style="float:left; margin-left:10px;" class="el-l-a-info">( type: <b>'+a.type+'</b> size: <b>'+a.size+'</b> )</div>'+
'        <div style="clear:both;"></div>'+
'        <div class="el-l-a-con el-l-a-hdn" id="el-l-a-con-'+a.id+'">'+
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
        	that.live_runs.push(thread_idx);
			if((that.live_min_run == 0) || (entry.run_num < that.live_min_run)) that.live_min_run = entry.run_num;
			if((that.live_max_run == 0) || (entry.run_num > that.live_max_run)) that.live_max_run = entry.run_num;
			html +=
'  <div class="el-l-r-hdr" id="el-l-m-hdr-'+thread_idx+'" onclick="elog.live_toggle_run('+thread_idx+');">'+
'    <div style="float:left;"><span class="ui-icon ui-icon-triangle-1-e el-l-r-tgl" id="el-l-r-tgl-'+entry.id+'"></span></div>'+
'    <div style="float:left;" class="el-l-m-time">'+entry.hms+'</div>'+
'    <div style="float:left;" class="el-l-m-author">'+entry.author+'</div>'+
'    <div style="float:left; margin-left:10px;" class="el-l-m-subj">'+entry.subject+'</div>'+
'    <div style="clear:both;"></div>'+
'  </div>'+
'  <div class="el-l-r-con el-l-r-hdn" id="el-l-r-con-'+entry.id+'"></div>';
		}
		return html;
	};

	this.live_child2html = function(entry) {
		that.live_messages[entry.id] = entry;
		var html =
'<div class="el-l-c-hdr" onclick="elog.live_toggle_child('+entry.id+');">'+
'  <div style="float:left;"><span class="ui-icon ui-icon-triangle-1-e el-l-c-tgl" id="el-l-c-tgl-'+entry.id+'"></span></div>'+
'  <div style="float:left;" class="el-l-c-time">'+entry.relevance_time+'</div>'+
'  <div style="float:left;" class="el-l-c-author">'+entry.author+'</div>'+
'  <div style="float:left; margin-left:10px;" class="el-l-c-subj">'+entry.subject+'</div>'+message_signs(entry)+
'  <div style="clear:both;"></div>'+
'</div>'+
'<div class="el-l-c-con el-l-c-hdn" id="el-l-c-con-'+entry.id+'">'+
'  <div class="el-l-c-body">'+
'    <div style="float:left; font-size:12px;">'+entry.html1+'</div>';
		if(that.editor)
			html +=
'    <div style="float:right;" class="s-b-con"><button class="el-l-m-ed"  id="el-l-m-ed-'+entry.id+'"  onclick="elog.live_message_edit('+entry.id+',true);">edit</button></div>';
		html +=
'    <div style="float:right;" class="s-b-con"><button class="el-l-m-re" id="el-l-m-re-'+entry.id+'" onclick="elog.live_message_reply('+entry.id+',true);">reply</button></div>'+
'    <div style="clear:both;"></div>'+
'      <div id="el-l-m-dlgs-'+entry.id+'"></div>'+
'  </div>';
		that.live_attachments[entry.id] = entry.attachments;

		var attachments_html = '';
		for( var i in entry.attachments) {
			var a = entry.attachments[i];
			that.live_attachments_loader[a.id] = {loaded: false, descr: a};
			attachments_html +=
'    <div style="float:left;" class="el-l-a">'+
'      <div style="float:left;">'+
'        <span class="ui-icon ui-icon-triangle-1-e el-l-a-tgl" id="el-l-a-tgl-'+a.id+'" onclick="elog.live_toggle_attachment('+a.id+');"></span>'+
'      </div>'+
'      <div style="float:left;" class="el-l-a-dsc"><a class="link" href="../logbook/attachments/'+a.id+'/preview"  target="_blank">'+a.description+'</a></div>'+
'      <div style="float:left; margin-left:10px;" class="el-l-a-info">( type: <b>'+a.type+'</b> size: <b>'+a.size+'</b> )</div>'+
'      <div style="clear:both;"></div>'+
'      <div class="el-l-a-con el-l-a-hdn" id="el-l-a-con-'+a.id+'">'+
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
	this.live_highlight_day = function(idx) {
		if(that.live_current_day == idx) return;
		that.live_current_day = idx;
		var day = that.live_days2threads[idx];
		applications['p-appl-elog'].context3 = '<a class="link" href="#el-l-m-d-hdr-'+idx+'" title="go to the day header">'+day.ymd+'</a>';
		set_context(applications['p-appl-elog']);
	};

	this.live_dim_day = function() {
		if(that.live_current_day == null) return;
		that.live_current_day = null;
		applications['p-appl-elog'].context3 ='';
		set_context(applications['p-appl-elog']);
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

	/* ----------------------------------
	 *  Application initialization point
	 * ----------------------------------
	 */
	this.init = function() {
		this.live_init();
		this.post_init();
		this.search_init();
	};
}

var elog = new elog_create();
