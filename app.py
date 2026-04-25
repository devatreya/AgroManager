from openreward.environments import Server

from env import AgroManager

server = Server(environments=[AgroManager])
app = server.app
