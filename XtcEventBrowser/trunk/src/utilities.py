class PyanaOptions( object ):
    def __init__( self ):
        pass

    def getOptStrings(self, options_string) :
        """
        parse the option string,
        return a tuple of N, item(s), i.e. item or list of items
        """
        if options_string is None:
            return None

        # make sure there are no newline characters here
        options_string = options_string.split("\n")
        options_string = " ".join(options_string)

        # make a list
        options = options_string.split(" ")

        if len(options)==0 :
            return []

        elif len(options)==1 :
            if ( options_string == "" or
                 options_string == "None" or
                 options_string == "No" ) :
                return []

        # all other cases:
        return options


    def getOptIntegers(self, options_string):
        """Return a list of integers
        """
        if options_string is None: return None

        opt = self.getOptStrings(options_string)
        N = len(opt)
        if N is 1:
            return int(opt)
        if N > 1 :
            items = []
            for item in opt :
                items.append( int(item) )
            return items
            

    def getOptInteger(self, options_string):
        """Return a single integer
        """
        if options_string is None: return None

        if options_string == "" : return None
        return int(options_string)


    def getOptBoolean(self, options_string):
        """Return a boolean
        """
        if options_string is None: return None

        opt = options_string
        if   opt == "False" or opt == "0" or opt == "No" or opt == "" : return False
        elif opt == "True" or opt == "1" or opt == "Yes" : return True
        else :
            print "utilities.getOptBoolean: cannot parse option ", opt
            return None

    def getOptBooleans(self, options_string):
        """Return a list of booleans
        """
        if options_string is None: return None

        opt_list = self.getOptStrings(options_string)
        N = len(opt_list)
        if N == 0 : return None
        
        opts = []
        for opt in optlist :
            opts.append( self.getOptBoolean(opt) )

        return opts


