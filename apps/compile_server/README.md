# Hidet Compilation Server

## Usage

### Setup the Compilation Server

```bash
$ # clone the hidet repository to the server
$ git clone https://github.com/hidet-org/hidet.git
$ cd hidet/apps/compile_server
$ # build the docker image and run it
$ bash run.sh
$ # Now, the compilation server is listening on port 3281
```

### Setup on the Client Side

```python
import hidet

# config the ip address and port of the server
hidet.option.compile_server.addr('x.x.x.x')  
hidet.option.compile_server.port(3281)      

# the username and password of the user, please change it to your own
hidet.option.compile_server.username('username')    
hidet.option.compile_server.password('password')

# the repository to use, by default, the main branch of hidet-org/hidet will be used
hidet.option.compile_server.repo('https://github.com/hidet-org/hidet', 'main')  

# enable the compile server
hidet.option.compile_server.enable()        
```
