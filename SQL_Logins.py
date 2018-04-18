import os

db_servers = {"LL-SQL-PI01": [os.environ.get("LL-SQL-PI01 USERNAME"),
                              os.environ.get("LL-SQL-PI01 PASSWORD")],

              "MHA-SQL-BW01.MHAITB.VPN": [os.environ.get("MHA-SQL-BW01.MHAITB.VPN USERNAME"),
                                          os.environ.get("MHA-SQL-BW01.MHAITB.VPN PASSWORD")],

              "www.lifeline-chat.org": [os.environ.get("www.lifeline-chat.org USERNAME"),
                                        os.environ.get("www.lifeline-chat.org PASSWORD")]}



