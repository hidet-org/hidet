# Hidet Compilation Server

## Usage

### Setup the Compilation Server

```bash
$ # clone the hidet repository to the server
$ git clone https://github.com/hidet-org/hidet.git
$ cd hidet/apps/compile_server
$ # build the docker image and run it
$ sudo docker build -t hidet-compile-server .
$ sudo docker run -d -p 3281:3281 hidet-compile-server
$ # Now, the compilation server is listening on port 3281
```

### Setup on the Client Side

```python
import hidet

# config the compile server
hidet.option.compile_server.addr('x.x.x.x')         # the ip address of the server
hidet.option.compile_server.port(3281)              # the port of the server
hidet.option.compile_server.username('username')    # the username and password of the user
hidet.option.compile_server.password('password')    # please change it to your own
hidet.option.compile_server.enable()                # enable the compile server
```
