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
