"""
USAGE:
You'll also want to monitor real-time changes to the PostgreSQL database server
you are running with Docker. Follow the instructions in the README.md of this
(sub-)project but otherwise, this is a summary:

Get the name and container ID of the Docker image that has postgresql running
docker ps
`exec` into the running Docker container, e.g.
docker exec -it local-llm-full-postgres psql -U inserviceofx -d local_llm_full_database
where you'll get the username and database name from the docker-compose.yml file.

You should see something like this:
psql (16.8 (Debian 16.8-1.pgdg120+1))
Type "help" for help.

Then you can do something like this:

# list all databases
\l
"""

from corecode.Utilities import DataSubdirectories, is_model_there
data_subdirectories = DataSubdirectories()
relative_model_path = "Models/Embeddings/BAAI/bge-large-en-v1.5"
is_model_downloaded, model_path = is_model_there(
    relative_model_path,
    data_subdirectories)
model_is_not_downloaded_message = f"Model {relative_model_path} not downloaded"

from pathlib import Path
from tools.Databases.PostgreSQLSetup import PostgreSQLSetupData, PostgreSQLSetup
import pytest

test_postgresql_setup_configuration_path = Path(__file__).parents[2] / \
    "TestSetup" / "postgresql_setup_configuration.yml"

test_postgresql_setup_data = PostgreSQLSetupData.from_yaml(
    test_postgresql_setup_configuration_path)

def test_PostgreSQLSetupData_from_yaml_works():
    assert test_postgresql_setup_data.database_port == 5432
    assert test_postgresql_setup_data.ip_address is not None
    assert test_postgresql_setup_data.postgres_user == "inserviceofx"
    assert test_postgresql_setup_data.postgres_password == "inserviceofx"
    assert test_postgresql_setup_data.database_names == \
        {"PermanentConversation": "test_permanent_conversation"}

@pytest.mark.asyncio
async def test_PostgreSQLSetup_creates_postgresql_connection_from_database_type():
    pgs_setup = PostgreSQLSetup(test_postgresql_setup_data)
    pgs_setup.create_postgresql_connection(database_type="PermanentConversation")
    database_name = test_postgresql_setup_data.database_names[
        "PermanentConversation"]

    assert await pgs_setup._connections["PermanentConversation"].database_exists(
        database_name) is False, \
        f"Database {database_name} should not exist yet!"

    await pgs_setup._connections["PermanentConversation"].create_database(
        database_name)

    assert await pgs_setup._connections["PermanentConversation"].database_exists(
        database_name) is True, \
        f"Database {database_name} should exist now!"

    await pgs_setup._connections["PermanentConversation"].drop_database(
        database_name)

    assert await pgs_setup._connections["PermanentConversation"].database_exists(
        database_name) is False, \
        f"Database {database_name} should not exist now!"

@pytest.mark.asyncio
async def test_PostgreSQLSetup_create_pool_for_database_creates():
    pgs_setup = PostgreSQLSetup(test_postgresql_setup_data)
    pgs_setup.create_postgresql_connection(database_type="PermanentConversation")
    database_name = test_postgresql_setup_data.database_names[
        "PermanentConversation"]

    assert await pgs_setup._connections["PermanentConversation"].database_exists(
        database_name) is False, \
        f"Database {database_name} should not exist yet!"

    await pgs_setup.create_pool_for_database(
        database_type="PermanentConversation")

    assert await pgs_setup._connections["PermanentConversation"].database_exists(
        database_name) is True, \
        f"Database {database_name} should exist now!"

    await pgs_setup._connections["PermanentConversation"].drop_database(
        database_name)

    assert await pgs_setup._connections["PermanentConversation"].database_exists(
        database_name) is False, \
        f"Database {database_name} should not exist now!"

@pytest.mark.asyncio
async def test_PostgreSQLSetup_create_pool_for_all_databases_creates():
    pgs_setup = PostgreSQLSetup(test_postgresql_setup_data)
    pgs_setup.create_connections_for_all_databases()
    await pgs_setup.create_pool_for_all_databases()

    for database_type in test_postgresql_setup_data.database_names:
        assert pgs_setup._connections[database_type] is not None, \
            f"Connection for {database_type} should be created"

        assert await pgs_setup._connections[database_type].database_exists(
            test_postgresql_setup_data.database_names[database_type]) is True, \
            f"Database {test_postgresql_setup_data.database_names[database_type]} should exist now!"

    for database_type in test_postgresql_setup_data.database_names:
        await pgs_setup._connections[database_type].drop_database(
            test_postgresql_setup_data.database_names[database_type])