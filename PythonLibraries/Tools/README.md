## PostgreSQL database: Basics

So you're running a Docker image that would then run, rely upon, docker-compose.yml to bring up a PostgreSQL, e.g.

```
InServiceOfX/Scripts/QuickAliases$ python QuickRunDocker.py --gpu 0 --build-dir LLM/LocalLLMFull
```

and you see this:

```
[+] Running 2/2
 ✔ Network local_llm_full_network     Created                                                                                      0.0s 
 ✔ Container local-llm-full-postgres  Started                                                                                      0.3s 
Using database network: local_llm_full_network
```

Great! Here are things you can do.

```
# Get the name and container ID of the Docker image that has postgresql running
docker ps
```
`exec` into the running Docker container, e.g.
```
docker exec -it local-llm-full-postgres psql -U inserviceofx -d local_llm_full_database

# You should see something like this:
psql (16.8 (Debian 16.8-1.pgdg120+1))
Type "help" for help.

local_llm_full_database=# 
```
`local-llm-full-postgres` is the name of the container, which would be specified in a `docker-compose.yml` file, in the field `container_name`. The PostgreSQL database name is also specified there, in the "field" `POSTGRES_DB`. In this case, the name is `local_llm_full_database`.


```
# list all databases
\l
```

*DO NOT DELETE* the database `postgre` if you see it in the list of databases; it's required for PostgreSQL to function properly. Likewise for template0, template1 databases.

### Deleting databases manually

```
DROP DATABASE test_pydantic_ai_database_1;
```
where I used the example of `test_pydantic_ai_database_1` (maybe spuriously created in a `pytest` integration test). Don't forget the semicolon at the end!