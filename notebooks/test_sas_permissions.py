# Databricks notebook source
# MAGIC %md
# MAGIC # Test SAS Token Permissions
# MAGIC 
# MAGIC Quick diagnostic to check what permissions are granted by Databricks temporary_path_credentials.

# COMMAND ----------
# MAGIC %pip install uv
# COMMAND ----------
# MAGIC %sh uv pip install -r ../requirements.lock

# COMMAND ----------
# MAGIC %restart_python

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

from azure.storage.filedatalake import FileSystemClient
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

# Initialize FileSystemClient with raw SAS token (not AzureSasCredential!)
# This is the key - using the token directly makes rename work
fs_client = FileSystemClient(account_url, container, credential=sas_token)

# COMMAND ----------

# Try rename with FileSystemClient (raw SAS token)
file_client = fs_client.get_file_client(source_path)

print("Source file client path:", file_client.path_name)
print("File URL:", file_client.url)

# The working format: container/full_path
new_name = f"{container}/{dest_path}"
print(f"\nAttempting rename to: {new_name}")

try:
    file_client.rename_file(new_name)
    print("✓ Rename succeeded!")
    # Clean up
    fs_client.get_file_client(dest_path).delete_file()
except Exception as e:
    print(f"✗ Rename failed: {e}")

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
# MAGIC ## Step 8: Benchmark SDK rename with GRIB-sized file

# COMMAND ----------

import time

# Create a ~38MB test file (similar to GRIB size)
GRIB_SIZE_MB = 38
large_source = f"{TEST_DIR}/large_source.bin"
large_dest = f"{TEST_DIR}/large_dest.bin"

print(f"Creating {GRIB_SIZE_MB}MB test file...")
with open(large_source, 'wb') as f:
    # Write in chunks to avoid memory issues
    chunk = b'x' * (1024 * 1024)  # 1MB chunk
    for _ in range(GRIB_SIZE_MB):
        f.write(chunk)
print(f"✓ Created {GRIB_SIZE_MB}MB file")

# COMMAND ----------

# Benchmark: SDK Rename (should work now with FileSystemClient)
large_source_path = f"{base_path}/_sas_test/large_source.bin"
large_dest_path = f"{base_path}/_sas_test/large_dest.bin"

source_client = fs_client.get_file_client(large_source_path)

print(f"\nBenchmarking SDK rename for {GRIB_SIZE_MB}MB file...")
print("=" * 50)

rename_start = time.perf_counter()
try:
    new_name = f"{container}/{large_dest_path}"
    source_client.rename_file(new_name)
    sdk_rename_time = time.perf_counter() - rename_start
    print(f"  SDK rename: {sdk_rename_time:.3f}s")
    sdk_rename_worked = True
except Exception as e:
    print(f"  SDK rename failed: {e}")
    sdk_rename_time = None
    sdk_rename_worked = False

# COMMAND ----------

# Compare with FUSE rename
print("\nBenchmarking os.replace() via FUSE mount...")
print("=" * 50)

# Recreate source file at large_source (the SDK rename moved it to large_dest)
with open(large_source, 'wb') as f:
    chunk = b'x' * (1024 * 1024)
    for _ in range(GRIB_SIZE_MB):
        f.write(chunk)

fuse_start = time.perf_counter()
os.replace(large_source, large_dest)
fuse_time = time.perf_counter() - fuse_start
print(f"  FUSE rename: {fuse_time:.3f}s")

# COMMAND ----------

# Summary comparison
print("\n" + "=" * 50)
print("COMPARISON SUMMARY")
print("=" * 50)
if sdk_rename_worked:
    print(f"  Azure SDK rename:  {sdk_rename_time:.3f}s")
else:
    print(f"  Azure SDK rename:  FAILED")
print(f"  FUSE os.replace(): {fuse_time:.3f}s")
print()
if sdk_rename_worked and sdk_rename_time < fuse_time:
    speedup = fuse_time / sdk_rename_time
    print(f"  → Azure SDK is {speedup:.1f}x FASTER")
elif sdk_rename_worked:
    slowdown = sdk_rename_time / fuse_time
    print(f"  → Azure SDK is {slowdown:.1f}x SLOWER")
else:
    print(f"  → SDK rename failed, FUSE is the only option")
print()
print(f"At 1 file/second rate:")
if sdk_rename_worked:
    print(f"  Azure SDK: {'✓ feasible' if sdk_rename_time < 1.0 else '✗ too slow'} ({sdk_rename_time:.3f}s per file)")
else:
    print(f"  Azure SDK: N/A (rename failed)")
print(f"  FUSE:      {'✓ feasible' if fuse_time < 1.0 else '✗ too slow'} ({fuse_time:.3f}s per file)")

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
# MAGIC **Key finding:** Using `FileSystemClient` directly with the raw SAS token (not `AzureSasCredential`) makes rename work!
# MAGIC
# MAGIC If SDK rename is faster than FUSE and under 1 second, we should use it for releasing files.

