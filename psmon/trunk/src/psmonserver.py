#!/usr/bin/env python
import os
import re
import zmq
import sys
import imp
import socket
import inspect
import logging
import argparse
import threading
import subprocess

from psmon import psapp, psconfig


LOG = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])


class ServerScript(object):
    def __init__(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.reset_socket = self.context.socket(zmq.REP)
        self.initialized = False
        self.__signal = re.compile(psconfig.RESET_REQ_STR%'(.*)')
        self.__reset_lock = threading.Lock()
        self.__reset_flag = threading.Event()

    def socket_init(self, port, reset_port, bufsize):
        self.socket.set_hwm(bufsize)
        self.socket.bind("tcp://*:%d" % port)
        self.socket.set_hwm(bufsize)
        self.reset_socket.bind("tcp://*:%d" % reset_port)
        self.initialized = True

    def send_data(self, topic, data):
        if self.initialized:
            self.socket.send(topic + psconfig.ZMQ_TOPIC_DELIM_CHAR, zmq.SNDMORE)
            self.socket.send_pyobj(data)

    def get_reset_flag(self):
        with self.__reset_lock:
            return self.__reset_flag.is_set()

    def set_reset_flag(self):
        with self.__reset_lock:
            self.__reset_flag.set()

    def clear_reset_flag(self):
        with self.__reset_lock:
            self.__reset_flag.clear()

    def reset_listener(self):
        while self.initialized:
            reset_msg = self.reset_socket.recv()
            signal_matcher = self.__signal.match(reset_msg)
            if signal_matcher is not None:
                self.set_reset_flag()
                self.reset_socket.send(psconfig.RESET_REP_STR%signal_matcher.group(1))
            else:
                self.reset_socket.send("invalid request from client")

    def run(self, expname):
        pass


def import_script(filename):
    path, name = os.path.split(filename)
    name, _ = os.path.splitext(name)

    fp, pathname, description = imp.find_module(name, [path])

    try:
        # add the scripts directory to the end of the pythonpath
        sys.path.append(path)
        return imp.load_module(name, fp, pathname, description)
    finally:
        if fp:
            fp.close()    


def parse_cmdline():
    # try to guess the default run
    default_exp, default_run = psapp.default_run_chooser() 

    parser = argparse.ArgumentParser(
        description='Psana plot server application'
    )

    parser.add_argument(
        'script',
        metavar='SCRIPT',
        help='The script to run on the server.'
    )

    parser.add_argument(
        'script_args',
        metavar='SCRIPT_ARGS',
        nargs='*',
        help='A list of arguments to pass to the script. Single values and key value pairs '\
             ' of the form <key>=<value>. Passed to the run function of the script as *arg '\
             'and **kwarg respectively. Example: foo bar baz=value'
    )

    parser.add_argument(
        '-p',
        '--port',
        metavar='PORT',
        type=int,
        default=psconfig.APP_PORT,
        help='the tcp port the server listens on (default: %d)'%psconfig.APP_PORT
    )

    parser.add_argument(
        '-b',
        '--buffer',
        metavar='BUFFER',
        type=int,
        default=psconfig.APP_BUFFER,
        help='the size in messages of send buffer (default: %d)'%psconfig.APP_BUFFER
    )

    parser.add_argument(
        '-e',
        '--expname',
        metavar='EXPNAME',
        default=default_exp,
        help='the experiment name or online (default: %s)'%default_exp
    )

    parser.add_argument(
        '-r',
        '--run',
        metavar='RUN',
        default=default_run,
        help='the run number (default: %s)'%default_run
    )

    parser.add_argument(
        '--log',
        metavar='LOG',
        default=psconfig.LOG_LEVEL,
        help='the logging level of the client (default %s)'%psconfig.LOG_LEVEL
    )

    return parser.parse_args()


def main():
    try:
        # initialize the logging system
        psapp.log_init()

        # grab the cli args
        args = parse_cmdline()

        # set levels for loggers that we care about
        LOG.setLevel(psapp.log_level_parse(args.log))

        # parse any arguments for the script
        script_args, script_kwargs = psapp.parse_args(*args.script_args)

        script_mod = import_script(args.script)

        def valid_class(obj):
            """Simple predicate function used to inspect the module"""
            return inspect.isclass(obj) \
                and script_mod.__name__ == obj.__module__
                #and issubclass(obj, ServerScript)
                #and script_mod.__name__ == obj.__module__ \
                #and issubclass(obj, ServerScript)

        class_list = inspect.getmembers(script_mod, valid_class)

        if len(class_list) != 1:
            LOG.critical('The chosen server script %s is invalid: It contains %d ServerScript classes', args.script, len(class_list))
            return 2

        LOG.info('Running the server script: \'%s\'', args.script)
        server_script = class_list[0][1]()
        server_script.socket_init(args.port, args.port+1, args.buffer)
        listener_thread = threading.Thread(target=server_script.reset_listener) #, args=(server_script,)
        listener_thread.daemon = True
        listener_thread.start()

        if ((args.run).lower() == 'online'):
            cmd = "grep psana /reg/g/pcds/dist/pds/%s/scripts/%s.cnf | grep -v '#' | head -n1 | awk {'print $3'} | sed s/_/-/g | tr -d '\n'"\
                % ((args.expname.lower(),) * 2)
            p = subprocess.Popen(cmd,stdout=subprocess.PIPE, shell=True)
            (onl_psana_node, errors) = p.communicate()
            hostname=socket.gethostname()
            if hostname != onl_psana_node:
                LOG.error('You are on: %s, please run the server on the psana shared memory node for %s: %s', hostname, args.expname, onl_psana_node)
                return 1
            else:
                p = subprocess.Popen("ls /dev/shm | grep psana | sed s/PdsMonitorSharedMemory_//g | tr -d '\n'",stdout=subprocess.PIPE, shell=True)
                (shmemname, errors) = p.communicate()
                expString = "shmem=%s.0:stop=no"%shmemname
                server_script.run(expString, *script_args, **script_kwargs)
        else:
            expString = "exp=%s:run=%s"%(args.expname, args.run)
            server_script.run(expString, *script_args, **script_kwargs)
        

    except (ImportError, SyntaxError) as imp_err:
        LOG.exception('The selected server script, %s, is invalid!', args.script)
        return 2
    except KeyboardInterrupt:
        print '\nExitting server!'


if __name__ == '__main__':
    sys.exit(main())
