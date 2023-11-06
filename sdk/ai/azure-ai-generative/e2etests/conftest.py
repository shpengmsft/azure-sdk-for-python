import logging
from pathlib import Path
import pytest
from typing import List
from azure.identity import AzureCliCredential, DefaultAzureCredential, InteractiveBrowserCredential

from azure.ai.resources.client import AIClient

logger = logging.getLogger(__name__)

from devtools_testutils import test_proxy

# autouse=True will trigger this fixture on each pytest run, even if it's not explicitly used by a test method
@pytest.fixture(scope="session", autouse=True)
def start_proxy(test_proxy):
    return


@pytest.fixture(scope="session")
def ubuntu_rag_environment(local_environments_base):
    from azure.ai.ml.entities import BuildContext, Environment

    return Environment(
        name="ubuntu_rag",
        description="AzureML RAG E2E Test Environment",
        build=BuildContext(path=local_environments_base / "ubuntu_rag"),
    )


@pytest.fixture(scope="session")
def local_azureml_rag_base():
    return Path(__file__).parent.parent.parent


def pytest_addoption(parser):
    from index.conftest import index_pytest_addoption

    parser.addoption("--subscription-id", default="b17253fa-f327-42d6-9686-f3e553e24763",
                     help="Subscription id of Azure AI Project used for testing.")
    parser.addoption("--resource-group", default="hanchi-test",
                     help="Resource group name of Azure AI Project used for testing.")
    parser.addoption("--project-name", default="hwep",
                     help="Name of Azure AI Project used for testing.")
    parser.addoption("--workspace-config-path", default="./config.json",
                     help="Path to workspace config file.")

    index_pytest_addoption(parser)


def pytest_generate_tests(metafunc):
    from index.conftest import pytest_generate_tests as index_generate_tests

    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test 'fixturenames'.
    args = ["subscription_id", "resource_group", "project_name", "workspace_config_path"]
    args.extend(index_generate_tests())
    parameterize_metafunc(metafunc, args)


def parameterize_metafunc(metafunc, param_names: List[str]):
    for param_name in param_names:
        if hasattr(metafunc.config.option, param_name):
            param_value = getattr(metafunc.config.option, param_name)
        else:
            param_value = None
        if param_name in metafunc.fixturenames:
            metafunc.parametrize(param_name, [param_value], scope="session")


@pytest.fixture(scope="session")
def azure_credentials():
    try:
        credential = AzureCliCredential(process_timeout=60)
        # Check if given credential can get token successfully.
        credential.get_token("https://management.azure.com/.default")
    except Exception:
        try:
            credential = DefaultAzureCredential(process_timeout=60)
            # Check if given credential can get token successfully.
            credential.get_token("https://management.azure.com/.default")
        except Exception:
            credential = InteractiveBrowserCredential()

    return credential


@pytest.fixture(scope="session")
def ai_client(azure_credentials, subscription_id, resource_group, project_name, workspace_config_path):
    if workspace_config_path is not None and len(workspace_config_path) > 0:
        logger.info(
            f"🔃 Loading Project details from config file: {workspace_config_path}.")
        try:
            return AIClient.from_config(credential=azure_credentials, path=workspace_config_path)
        except Exception as e:
            logger.warning(
                f"Failed to load Project details from config file: {workspace_config_path}: {e}")

    ai_client = AIClient(
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        ai_resource_name = project_name,
        project_name=project_name,
        credential=azure_credentials
    )
    logger.info(f"🔃 Using AIClient: {ai_client}")
    return ai_client


@pytest.fixture(scope="session")
def ml_client(ai_client):
    return ai_client._ml_client


@pytest.fixture(scope="session")
def azureml_workspace_v1(subscription_id, resource_group, project_name, workspace_config_path):
    from azureml.core import Workspace
    from azureml.core.authentication import AzureCliAuthentication

    if workspace_config_path is not None and len(workspace_config_path) > 0:
        logger.info(
            f"🔃 Loading workspace from config file: {workspace_config_path}.")
        try:
            return Workspace.from_config(path=workspace_config_path)
        except Exception as e:
            logger.warning(
                f"Failed to load workspace from config file: {workspace_config_path}: {e}")

    logger.info(f"🔃 Using AzureML workspace: {project_name}")
    return Workspace(subscription_id, resource_group, workspace_name=project_name, auth=AzureCliAuthentication())


@pytest.fixture(scope="session")
def azureml_workspace_v2(azure_credentials, subscription_id, resource_group, project_name, workspace_config_path):
    from azure.ai.ml import MLClient

    if workspace_config_path is not None and len(workspace_config_path) > 0:
        logger.info(
            f"🔃 Loading workspace from config file: {workspace_config_path}.")
        try:
            return MLClient.from_config(credential=azure_credentials, path=workspace_config_path)
        except Exception as e:
            logger.warning(
                f"Failed to load workspace from config file: {workspace_config_path}: {e}")

    logger.info(f"🔃 Using AzureML workspace: {project_name}")
    return MLClient(
        credential=azure_credentials,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=project_name
    )


@pytest.fixture()
def test_dir():
    test_dir = Path(__file__).parent
    logger.info(f"test directory is {test_dir}")
    return test_dir
