from pathlib import Path
from warnings import warn
import asyncio
import sys

common_api_path = Path(__file__).resolve().parents[5]

print("common_api_path", common_api_path)

if common_api_path.exists() and str(common_api_path) not in sys.path:
    sys.path.append(str(common_api_path))
elif not common_api_path.exists():
    warn(f"CommonAPI path {common_api_path} does not exist")

common_api_tests_path = common_api_path / "tests"

if common_api_tests_path.exists() \
    and str(common_api_tests_path) not in sys.path:
    sys.path.append(str(common_api_tests_path))

from commonapi.Databases.Messages.PermanentConversation \
    import PostgreSQLInterface as PermanentConversationPostgreSQLInterface

from commonapi.Databases import PostgreSQLConnection

from TestSetup.PostgreSQLDatabaseSetup import PostgreSQLDatabaseSetupData

async def main_function():
    test_database_name = "test_permanent_conversation_database"
    test_dsn = PostgreSQLDatabaseSetupData().test_dsn

    postgres_connection = PostgreSQLConnection(test_dsn, test_database_name)
    await postgres_connection.create_database(database_name=test_database_name)
    await postgres_connection.create_new_pool()
    await postgres_connection.create_extension("vector")

    permanent_conversation_postgres_interface = \
        PermanentConversationPostgreSQLInterface(postgres_connection)

    await permanent_conversation_postgres_interface.create_table()

    await postgres_connection.close()

if __name__ == "__main__":
    asyncio.run(main_function())