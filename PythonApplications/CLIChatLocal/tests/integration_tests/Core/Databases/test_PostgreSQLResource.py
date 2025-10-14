from clichatlocal import ApplicationPaths
from clichatlocal.Configuration.CLIConfiguration import CLIConfiguration
from clichatlocal.Core import ProcessConfigurations
from clichatlocal.Core.Databases import PostgreSQLResource
from clichatlocal.Terminal import TerminalUI
from tools.Databases.PostgreSQLSetup import PostgreSQLSetup

import pytest

@pytest.mark.asyncio
async def test_PostgreSQLResource_explicit_step_by_step():
    application_paths = ApplicationPaths.create(
        is_development=True,
        is_current_path=False,
        configpath=None)

    cli_configuration = CLIConfiguration()
    terminal_ui = TerminalUI(cli_configuration)

    process_configurations = ProcessConfigurations(
        application_paths,
        terminal_ui)
    process_configurations.process_configurations()

    postgresql_resource = PostgreSQLResource(process_configurations)

    assert postgresql_resource._pgs_setup is None

    postgresql_resource._pgs_setup = PostgreSQLSetup(
        postgresql_resource._load_setup_configuration())

    postgresql_resource._pgs_setup.create_connections_for_all_databases()

    assert len(postgresql_resource._pgs_setup._connections) == 1
    connection = postgresql_resource._pgs_setup._connections[
        "PermanentConversation"]
    assert connection is not None
    assert connection._database_name == "permanent_conversation"
    assert connection._server_data_source_name is not None
    # print(connection._server_data_source_name)
    assert connection._pool is None

    await postgresql_resource._pgs_setup.create_pool_for_all_databases()

    assert connection._pool is not None

    assert connection == postgresql_resource.get_connection(
        "PermanentConversation")

@pytest.mark.asyncio
async def test_PostgreSQLResource_load_configuration_and_create_pool_works():
    application_paths = ApplicationPaths.create(
        is_development=True,
        is_current_path=False,
        configpath=None)

    cli_configuration = CLIConfiguration()
    terminal_ui = TerminalUI(cli_configuration)

    process_configurations = ProcessConfigurations(
        application_paths,
        terminal_ui)
    process_configurations.process_configurations()

    postgresql_resource = PostgreSQLResource(process_configurations)

    assert postgresql_resource._pgs_setup is None

    await postgresql_resource.load_configuration_and_create_pool()

    assert len(postgresql_resource._pgs_setup._connections) == 1
    connection = postgresql_resource._pgs_setup._connections[
        "PermanentConversation"]
    assert connection is not None
    assert connection._database_name == "permanent_conversation"
    assert connection._server_data_source_name is not None
    assert connection._pool is not None

    assert connection == postgresql_resource.get_connection(
        "PermanentConversation")
