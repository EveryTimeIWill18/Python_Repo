from login import *
import tableauserverclient as TSC


def login(f):
    def wrapper(srvr, un, pw, *args, **kwargs):
        """login to tableau server"""
        server = TSC.Server(srvr)
        auth = TSC.TableauAuth(un, pw)
        login = server.auth.sign_in(auth)
        tableau_item = f(*args, **kwargs)
        return login
    return wrapper


class Login(object):
    """Tableau Server"""

    def __init__(self):
        self._server_information = dict()
        self._is_logged_in = False

    def __getattr__(self, inval):
        print("Attribute: {}, is not valid Tableau call.".__format__(inval))

    def login(self, srvr, un, pw):
        """login to server"""
        server = TSC.Server(srvr)
        print(type(server))
        auth = TSC.TableauAuth(un, pw)
        print(type(auth))
        self._server_information.update({"Server": [srvr, server]})
        self._server_information.update({"USER": un})
        print(type(server.auth.sign_in(auth)))




nspl_ = Login()

nspl_.login(srvr=nspl_tableau_server,
            un=nspl_tableau_username,
            pw=nspl_tableau_password)




