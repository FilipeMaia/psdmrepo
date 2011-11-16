function p_appl_admin() {
	var that = this;
	this.name = 'admin';
	this.full_name = 'Admin';
	this.context = '';
	this.select = function(ctx) {
		that.context = ctx;
	};
	this.select_default = function() {
		if( this.context == '' ) this.context = 'cablenumbers';
	};
	this.if_ready2giveup = function( handler2call ) {
		handler2call();
	};
	return this;
}
var admin = new p_appl_admin();


function p_appl_search() {
	var that = this;
	this.name = 'search';
	this.full_name = 'Search';
	this.context = '';
	this.when_done = null;
	this.select = function(ctx, when_done) {
		that.context = ctx;
		this.when_done = when_done;
		this.init();
	};
	this.select_default = function() {
		if( this.context == '' ) this.context = 'cables';
		this.init();
	};
	this.search_cables = function() {
		$('#search-cables-search').button('disable');
		$('#search-cables-reset').button('disable');
		$('#search-cables-info').html('Searching...');
		var params = { is_submitted: '' };
		var device_name = $('#search-cables-form input[name="device_name"]').val();
		if(device_name != '') params.device_name = device_name;
		var jqXHR = $.get(
			'../portal/neocaptar_search_cables.php', params,
			function(data) {
				$('#search-cables-info').html('Found <b>123</b> entries');
				$('#search-cables-result').html(data);
			},
			'HTML'
		).error(
			function () {
				$('#search-cables-info').html('failed because of: '+jqXHR.statusText);
			}
		).complete(
			function () {
				$('#search-cables-search').button('enable');
				$('#search-cables-reset').button('enable');
			}
		);
	};
	this.initialized = false;
	this.init = function() {
		if( this.initialized ) return;
		this.initialized = true;
		$('#search-cables-search').button().click(function() { that.search_cables(); });
		$('#search-cables-reset' ).button().click(function() {});
	};
	this.if_ready2giveup = function( handler2call ) {
		handler2call();
	};
	return this;
}
var search = new p_appl_search();
