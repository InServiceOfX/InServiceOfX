# Reasons for Common API

I see this message format alot:

```
    messages=[

        {

            "role": "system",

            "content": "You are a calculator assistant. Use the calculate function to perform mathematical operations and provide the results."

        },

        {

            "role": "user",

            "content": user_prompt,

        }

    ]
```
Let's make a wrapper for this since we see this repeated all the time.

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

```
# list all databases
\l
```