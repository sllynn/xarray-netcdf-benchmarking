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

# Try the rename operation with different path formats
service_client = DataLakeServiceClient(account_url, credential=AzureSasCredential(sas_token))
fs_client = service_client.get_file_system_client(container)
file_client = fs_client.get_file_client(source_path)

print("Source file client path:", file_client.path_name)
print("File URL:", file_client.url)

# Try multiple destination path formats
dest_formats = [
    ("container/full_path", f"{container}/{dest_path}"),
    ("full_path_only", dest_path),
    ("relative_to_base", "_sas_test/test_dest.txt"),
    ("just_filename", "test_dest.txt"),
]

for format_name, dest in dest_formats:
    # Recreate source file for each test
    with open(test_file_1, 'w') as f:
        f.write("test content")
    
    print(f"\nAttempting rename with format '{format_name}':")
    print(f"  Destination: {dest}")
    
    # Get fresh file client
    file_client = fs_client.get_file_client(source_path)
    
    try:
        file_client.rename_file(dest)
        print(f"  ✓ Rename succeeded with format: {format_name}")
        # Clean up dest file
        try:
            fs_client.get_file_client(dest_path).delete_file()
        except:
            pass
        break
    except Exception as e:
        error_code = getattr(e, 'error_code', 'unknown')
        print(f"  ✗ Failed: {error_code}")
        if "AuthorizationPermissionMismatch" not in str(e):
            print(f"    {e}")

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
# MAGIC ## Step 6: Test Delete separately

# COMMAND ----------

# Recreate source file
with open(test_file_1, 'w') as f:
    f.write("test content for delete test")

delete_client = fs_client.get_file_client(source_path)

print("Testing delete via Azure SDK...")
try:
    delete_client.delete_file()
    print("✓ Delete succeeded!")
except Exception as e:
    print(f"✗ Delete failed: {type(e).__name__}: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Test Copy + Delete pattern

# COMMAND ----------

# Recreate source file via FUSE
with open(test_file_1, 'w') as f:
    f.write("test content for copy test")

print("Testing copy operation via Azure SDK...")
source_client = fs_client.get_file_client(source_path)

# Read source content
try:
    download = source_client.download_file()
    content = download.readall()
    print(f"  ✓ Read source: {len(content)} bytes")
except Exception as e:
    print(f"  ✗ Read failed: {e}")
    content = None

if content:
    # Write to destination
    dest_client = fs_client.get_file_client(dest_path)
    try:
        dest_client.create_file()
        dest_client.append_data(content, 0)
        dest_client.flush_data(len(content))
        print("  ✓ Write to dest succeeded!")
        
        # Now try to delete source
        try:
            source_client.delete_file()
            print("  ✓ Delete source succeeded!")
            print("\n✓✓✓ COPY + DELETE pattern works! ✓✓✓")
        except Exception as e:
            print(f"  ✗ Delete source failed: {e}")
    except Exception as e:
        print(f"  ✗ Write to dest failed: {e}")

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
# MAGIC **If rename fails but copy+delete works:**
# MAGIC - The SAS token has the right permissions, but the `rename_file()` API has specific requirements
# MAGIC - We can use copy+delete as an alternative (might be slightly slower but should still be fast for small files)
# MAGIC
# MAGIC **If both fail:**
# MAGIC - There may be path scoping issues with how the SAS token is generated
# MAGIC - Check the exact error messages for clues

