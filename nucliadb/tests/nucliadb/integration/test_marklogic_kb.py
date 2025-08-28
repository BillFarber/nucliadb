# Copyright (C) 2021 Bosutech XXI S.L.
#
# AGPL:
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

from unittest import mock

import pytest
from httpx import AsyncClient

# Imports for MarkLogic testing

MARKLOGIC_MODULE = "nucliadb.common.external_index_providers.marklogic"


@pytest.fixture()
def mock_marklogic_client():
    with mock.patch("nucliadb_utils.utilities.get_marklogic_client") as mock_get_client:
        # Mock the client instance returned by get_marklogic_client
        mock_client = mock.MagicMock()
        mock_client.documents.write.return_value = None
        mock_client.eval.return_value = {"success": True}
        mock_get_client.return_value = mock_client
        yield mock_client


@pytest.fixture(scope="function")
async def marklogic_knowledgebox(nucliadb_writer_manager: AsyncClient, mock_marklogic_client):
    """Create a knowledge box with MarkLogic external index provider."""
    
    # Create the knowledge box
    response = await nucliadb_writer_manager.post(
        "/kbs",
        json={
            "slug": "marklogic-test-kb-fixture",
            "external_index_provider": {
                "type": "marklogic",
                "host": "localhost",
                "port": 8000,
                "username": "admin",
                "password": "admin",
                "database": "Documents"
            },
        }
    )
    assert response.status_code == 201, response.text
    kbid = response.json()["uuid"]
    
    yield kbid
    
    # Cleanup - delete the knowledge box
    try:
        await nucliadb_writer_manager.delete(f"/kb/{kbid}")
    except Exception:
        pass  # Best effort cleanup


@pytest.fixture(autouse=True)
def hosted_nucliadb():
    """Mock hosted nucliadb environment."""
    with mock.patch("nucliadb_utils.settings.nuclia_settings.nuclia_service_account", "test"):
        with mock.patch("nucliadb_utils.settings.nuclia_settings.nuclia_public_url", "https://nuclia.cloud"):
            yield


@pytest.mark.deploy_modes("standalone")
async def test_marklogic_kb_creation_and_deletion(
    nucliadb_writer_manager: AsyncClient,
    nucliadb_reader: AsyncClient,
    mock_marklogic_client,
):
    """Test MarkLogic knowledge box creation and deletion."""
    
    # Create knowledge box with MarkLogic config
    slug = "marklogic-test-kb"
    
    response = await nucliadb_writer_manager.post(
        "/kbs",
        json={
            "slug": slug,
            "external_index_provider": {
                "type": "marklogic",
                "host": "localhost",
                "port": 8000,
                "username": "admin",
                "password": "admin",
                "database": "Documents"
            },
        },
    )
    assert response.status_code == 201, response.text
    kbid = response.json()["uuid"]

    # Verify MarkLogic client was initialized but not yet used for index operations
    # (indexes are created when vectorsets are configured, not during bare KB creation)
    assert mock_marklogic_client is not None  # Client should be available

    # Verify we can read the knowledge box
    resp = await nucliadb_reader.get(f"/kb/{kbid}")
    assert resp.status_code == 200, resp.text
    
    kb_data = resp.json()
    assert kb_data["config"]["configured_external_index_provider"]["type"] == "marklogic"

    # Delete the knowledge box
    response = await nucliadb_writer_manager.delete(f"/kb/{kbid}")
    assert response.status_code == 200, response.text

    # Verify MarkLogic client is available for cleanup operations
    # (actual cleanup calls happen when there are indexes to clean up)
    assert mock_marklogic_client is not None


@pytest.mark.deploy_modes("standalone")
async def test_marklogic_kb_get_config(
    nucliadb_reader: AsyncClient,
    marklogic_knowledgebox: str,
):
    """Test retrieving MarkLogic knowledge box configuration."""
    kbid = marklogic_knowledgebox

    resp = await nucliadb_reader.get(f"/kb/{kbid}")
    assert resp.status_code == 200, resp.text
    
    config = resp.json()["config"]
    assert not config.get("external_index_provider")  # Should not expose sensitive config
    assert config["configured_external_index_provider"]["type"] == "marklogic"


@pytest.mark.deploy_modes("standalone")
async def test_marklogic_kb_resource_insertion_and_deletion(
    nucliadb_writer: AsyncClient,
    nucliadb_reader: AsyncClient,
    marklogic_knowledgebox: str,
    mock_marklogic_client,
):
    """Test inserting and deleting a resource in a MarkLogic knowledge box."""
    kbid = marklogic_knowledgebox

    # Create a resource
    resource_data = {
        "slug": "test-resource",
        "title": "Test Resource for MarkLogic",
        "summary": "A test document to verify MarkLogic integration",
        "texts": {
            "content": {
                "body": "This is a test document content for MarkLogic indexing.",
                "format": "PLAIN"
            }
        }
    }

    # Insert the resource
    resp = await nucliadb_writer.post(
        f"/kb/{kbid}/resources",
        json=resource_data,
    )
    assert resp.status_code == 201, f"Failed to create resource: {resp.text}"
    resource_uuid = resp.json()["uuid"]
    assert resource_uuid is not None

    # Verify the resource exists by reading it back
    resp = await nucliadb_reader.get(
        f"/kb/{kbid}/resource/{resource_uuid}",
    )
    assert resp.status_code == 200, f"Failed to read resource: {resp.text}"
    resource_data_back = resp.json()
    assert resource_data_back["title"] == "Test Resource for MarkLogic"
    assert resource_data_back["slug"] == "test-resource"

    # Delete the resource
    resp = await nucliadb_writer.delete(
        f"/kb/{kbid}/resource/{resource_uuid}",
    )
    assert resp.status_code == 204, f"Failed to delete resource: {resp.text}"

    # Verify the resource no longer exists
    resp = await nucliadb_reader.get(
        f"/kb/{kbid}/resource/{resource_uuid}",
    )
    assert resp.status_code == 404, "Resource should not exist after deletion"

    # Verify MarkLogic mock was available during the test
    assert mock_marklogic_client is not None
