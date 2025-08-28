# Copyright (C) 2021 Bosutech XXI S.L.
#
# nucliadb is offered under the AGPL v3.0 and as commercial software.
# For commercial licensing, contact us at info@nuclia.com.
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

# Top-level conftest.py that consolidates all pytest_plugins from subprojects
# This is required by modern pytest versions that no longer support pytest_plugins
# in non-top-level conftest files.

pytest_plugins = []

# Core pytest plugins
pytest_plugins.extend([
    "pytest_mock",
    "pytest_docker_fixtures",
])

# NucliaDB main fixtures (only loaded when running nucliadb tests specifically)
# These are conditional to avoid import errors when running tests from other packages
try:
    import tests.ndbfixtures.magic  # Test if the module is available
    pytest_plugins.extend([
        # pytest hacks and magic to implement deploy_mode parametrization
        "tests.ndbfixtures.magic",
        # components and its deploy modes
        "tests.ndbfixtures.standalone",
        "tests.ndbfixtures.reader",
        "tests.ndbfixtures.writer",
        "tests.ndbfixtures.search",
        "tests.ndbfixtures.train",
        "tests.ndbfixtures.ingest",
        # subcomponents
        "tests.ndbfixtures.common",
        "tests.ndbfixtures.maindb",
        "tests.ndbfixtures.nidx",
        "tests.ndbfixtures.processing",
        # useful resources for tests (KBs, resources, ...)
        "tests.ndbfixtures.resources",
        "tests.nucliadb.knowledgeboxes",
        # legacy fixtures waiting for a better place
        "tests.ndbfixtures.legacy",
        # Legacy fixture from ingest
        "tests.ingest.fixtures",
    ])
except ImportError:
    # These fixtures are only available when running from within the nucliadb package
    pass

# Fixture from subpackages
pytest_plugins.extend([
    "nucliadb_utils.tests.fixtures",
    "nucliadb_utils.tests.nats",
    "nucliadb_utils.tests.gcs",
    "nucliadb_utils.tests.s3",
    "nucliadb_utils.tests.azure",
    "nucliadb_utils.tests.local",
    "nucliadb_utils.tests.asyncbenchmark",
])

# SDK fixtures
try:
    import nucliadb_sdk.tests.fixtures
    pytest_plugins.extend([
        "nucliadb_sdk.tests.fixtures",
    ])
except ImportError:
    pass

# Dataset fixtures
try:
    import nucliadb_dataset.tests.fixtures
    pytest_plugins.extend([
        "nucliadb_dataset.tests.fixtures",
    ])
except ImportError:
    pass

# Telemetry fixtures  
try:
    import nucliadb_telemetry.tests.telemetry
    pytest_plugins.extend([
        "nucliadb_telemetry.tests.telemetry",
    ])
except ImportError:
    pass
