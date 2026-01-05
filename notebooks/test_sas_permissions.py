# Databricks notebook source
# MAGIC %md
# MAGIC # Test SAS Token Permissions
# MAGIC 
# MAGIC Quick diagnostic to check what permissions are granted by Databricks temporary_path_credentials.

# COMMAND ----------

# Configuration - adjust to match your setup
CATALOG = "stuart"
SCHEMA = "lseg"
VOLUME_NAME = "netcdf"

VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}"
TEST_DIR = f"{VOLUME_PATH}/_sas_test"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Get SAS Token from Databricks

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import PathOperation

w = WorkspaceClient()

# Get volume storage location
volume_info = w.volumes.read(f"{CATALOG}.{SCHEMA}.{VOLUME_NAME}")
volume_root_url = volume_info.storage_location
print(f"Volume storage URL: {volume_root_url}")

# Generate temporary credentials
creds = w.temporary_path_credentials.generate_temporary_path_credentials(
    url=volume_root_url,
    operation=PathOperation.PATH_READ_WRITE,
)

sas_token = creds.azure_user_delegation_sas.sas_token
print(f"SAS Token (first 50 chars): {sas_token[:50]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Parse SAS Token Permissions

# COMMAND ----------

from urllib.parse import parse_qs

# Parse the SAS token to see what permissions it grants
sas_params = parse_qs(sas_token)

print("SAS Token Parameters:")
print("=" * 50)

# Key parameters to look for
param_descriptions = {
    'sp': 'Signed Permissions',
    'sr': 'Signed Resource',
    'se': 'Signed Expiry',
    'st': 'Signed Start',
    'spr': 'Signed Protocol',
    'sv': 'Signed Version',
    'skoid': 'Signed Key Object ID',
    'sktid': 'Signed Key Tenant ID',
}

for key, desc in param_descriptions.items():
    if key in sas_params:
        print(f"  {desc} ({key}): {sas_params[key][0]}")

# Permission breakdown
if 'sp' in sas_params:
    perms = sas_params['sp'][0]
    print("\nPermission Breakdown:")
    print("=" * 50)
    perm_map = {
        'r': 'Read',
        'w': 'Write', 
        'd': 'Delete',
        'l': 'List',
        'a': 'Add',
        'c': 'Create',
        'u': 'Update',
        'p': 'Process',
        'x': 'Execute',
        'o': 'Ownership',
        'm': 'Move',
        'e': 'Execute (POSIX)',
        't': 'Tag',
        'i': 'Set Immutability Policy',
    }
    
    for char in perms:
        desc = perm_map.get(char, f'Unknown ({char})')
        print(f"  ✓ {desc} ({char})")
    
    # Check for missing critical permissions
    print("\nMissing Permissions (needed for rename):")
    print("=" * 50)
    needed = {'d': 'Delete', 'm': 'Move'}
    for char, desc in needed.items():
        if char not in perms:
            print(f"  ✗ {desc} ({char}) - MISSING")
        else:
            print(f"  ✓ {desc} ({char}) - present")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Test Actual Operations

# COMMAND ----------

import os

# Create test directory via FUSE (should work)
os.makedirs(TEST_DIR, exist_ok=True)

# Create a test file via FUSE
test_file_1 = f"{TEST_DIR}/test_source.txt"
test_file_2 = f"{TEST_DIR}/test_dest.txt"

with open(test_file_1, 'w') as f:
    f.write("test content")

print(f"✓ Created test file via FUSE: {test_file_1}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Test Azure SDK Rename

# COMMAND ----------

from azure.storage.filedatalake import DataLakeServiceClient
from azure.core.credentials import AzureSasCredential
import re

# Parse storage URL
match = re.match(r'abfss://([^@]+)@([^.]+)\.dfs\.core\.windows\.net/(.+)', volume_root_url)
if not match:
    raise ValueError(f"Cannot parse storage URL: {volume_root_url}")

container = match.group(1)
account = match.group(2)
base_path = match.group(3)

account_url = f"https://{account}.dfs.core.windows.net"
print(f"Account URL: {account_url}")
print(f"Container: {container}")
print(f"Base path: {base_path}")

# Build paths relative to container root
source_path = f"{base_path}/_sas_test/test_source.txt"
dest_path = f"{base_path}/_sas_test/test_dest.txt"
print(f"Source path: {source_path}")
print(f"Dest path: {dest_path}")

# COMMAND ----------

# Try the rename operation
service_client = DataLakeServiceClient(account_url, credential=AzureSasCredential(sas_token))
fs_client = service_client.get_file_system_client(container)
file_client = fs_client.get_file_client(source_path)

print("Attempting rename via Azure SDK...")
try:
    file_client.rename_file(f"{container}/{dest_path}")
    print("✓ Rename succeeded!")
except Exception as e:
    print(f"✗ Rename failed: {type(e).__name__}")
    print(f"  Error: {e}")
    
    # Check if it's a permission error
    error_str = str(e)
    if "AuthorizationPermissionMismatch" in error_str:
        print("\n" + "=" * 50)
        print("CONFIRMED: SAS token lacks required permissions for rename")
        print("The 'd' (Delete) permission is required but not granted")
        print("=" * 50)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Test Write (should work)

# COMMAND ----------

# Test that write operations work (they should)
write_test_path = f"{base_path}/_sas_test/sdk_write_test.txt"
write_client = fs_client.get_file_client(write_test_path)

print("Testing write via Azure SDK...")
try:
    write_client.create_file()
    write_client.append_data(b"test content via SDK", 0)
    write_client.flush_data(len(b"test content via SDK"))
    print("✓ Write succeeded!")
except Exception as e:
    print(f"✗ Write failed: {type(e).__name__}: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cleanup

# COMMAND ----------

import shutil
try:
    shutil.rmtree(TEST_DIR)
    print(f"✓ Cleaned up {TEST_DIR}")
except Exception as e:
    print(f"Cleanup warning: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC If you see `AuthorizationPermissionMismatch` on the rename operation, it confirms that:
# MAGIC 1. Databricks `temporary_path_credentials` with `PATH_READ_WRITE` does NOT grant Delete permission
# MAGIC 2. Azure Data Lake rename requires Delete permission on the source file
# MAGIC 3. We need an alternative approach for fast file releases

