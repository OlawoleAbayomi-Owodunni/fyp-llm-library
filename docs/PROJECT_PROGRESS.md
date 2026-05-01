# Project Progress Log

Last updated: 2026-04-30

This document tracks what has been completed for the Cloud Data Centre assignment and the current state of the project after each step. It is meant to make the report easier to write later.

## Overall Goal

Build a small cloud-deployable application based on the existing LLM wrapper repository, using a simple web endpoint around the C++ inference binary.

## Step Status

### Step 1 - Update `LLMTest.cpp` to accept a prompt

Status: Complete

What was done:
- Updated `llm/tests/LLMTest.cpp` so the executable accepts a prompt from `argv`.
- Kept a default prompt for local testing when no argument is supplied.
- Printed the generated output to stdout so it can be consumed by a wrapper service.

Current state:
- `LLMTest` can now be run from the command line with a custom prompt.
- The executable is usable as the core generation backend for later steps.

### Step 2 - Build and verify the project locally

Status: Complete

What was done:
- Built the project successfully with CMake.
- Verified that the `LLMTest` executable runs locally and generates a response when given a quoted prompt.

Current state:
- The C++ build is working end-to-end.
- The repo has a working local generation executable that can be tested from the terminal.

### Step 3 - Add a lightweight Python web wrapper

Status: Complete

What was done:
- Added a Flask-based web layer in `web/api.py`.
- The web app calls the `LLMTest` executable through `subprocess`.
- The endpoint returns JSON for easy API testing.
- Added `web/requirements.txt` for the Python dependencies.

Current state:
- The project now has a simple HTTP interface over the C++ generator.
- This is the first cloud-ready service shape for the assignment.

### Step 4 - Test the wrapped service locally

Status: Complete

What was done:
- Started the Flask app locally.
- Sent POST requests to `/generate` with `curl.exe`.
- Confirmed the API returned JSON containing the original prompt and generated response.
- Measured end-to-end request time for three runs.

Observed timings:
- Run 1: 6.054435s
- Run 2: 6.117496s
- Run 3: 6.099680s

Current state:
- The web wrapper works locally and returns a valid JSON response.
- The request time is consistently around 6.1 seconds for this prompt and model setup.
- The project is ready to move on to cloud setup and deployment planning.

### Step 5 - Create Azure resources and upload the model

Status: Complete

What was done:
- Created an Azure resource group named `llm-lib-rg`.
- Created an Azure storage account named `llmlibstorage26`.
- Created a private blob container for model storage.
- Uploaded the GGUF model file `Llama-3.2-1B-Instruct-Q4_K_M.gguf`.

Observed storage details:
- Blob file size: 770.28 MiB

Current state:
- The cloud storage layer is now ready for deployment.
- The model is stored in Azure Blob Storage and can be downloaded later by the VM setup process.
- The storage provisioning part of the cloud setup is complete.


### Step 6 - Create the Azure VM and open port 5000

Status: Complete

What was done:
- Created an Azure virtual machine (`llm-lib-vm`) in the resource group `llm-lib-rg`.
- Selected Ubuntu Server 24.04 LTS as the image and a small general-purpose VM size.
- Configured the VM with a public IP address.
- Added an inbound Network Security Group rule allowing TCP port `5000` with priority `400` (rule name: `allow-flask-5000`).

Observed VM details:
- Public IP address: 20.203.185.178
- NSG inbound rule priority: 400

Current state:
- The VM exists in the same region as the storage account and has a public IP.
- Port 5000 is allowed through the VM's network security group so the Flask app can be reached externally.
- Next step: SSH into the VM and perform the setup/build steps so the service can run on the VM.

### Step 7 - SSH to the VM and run setup

Status: Complete

What was done:
- SSH'd into the VM (`llm-lib-vm`) and installed build tools and runtime dependencies.
- Cloned the repository onto the VM and downloaded the model from Azure Blob Storage into `resources/downloaded_resources/` using a SAS URL.
- Built `LLMTest` with CMake on the VM.
- Created a Python virtual environment in `web/.venv`, installed `Flask` and `gunicorn`, and started the web service with `gunicorn`.

Observed remote timings (three runs via `curl` from local machine):
- Run 1: 19.152031s
- Run 2: 19.342630s
- Run 3: 19.436745s

Server evidence:
- `gunicorn` was started on the VM and reported listening at `http://0.0.0.0:5000` and booted a worker process (logs captured).

Current state:
- The VM is running the Flask service behind `gunicorn` and is responding to external requests.
- Cold-start + generation time observed remotely is ~19.2–19.4 seconds for this prompt/model on the VM (recorded above).
- Next step: verify endpoint and capture screenshots/logs for documentation and run multiple warm requests to collect CPU/memory stats while generating.

### Step 8 - Verify remote endpoint and capture evidence

Status: Complete

What was done:
- Verified the remote `/generate` endpoint by sending multiple `curl` requests from the local machine.
- Collected response JSON files.
- Captured `gunicorn` logs (access/error) from the VM and copied them locally (`gunicorn-access.log`, `gunicorn-error.log`).
- Collected CPU/memory snapshots during generation (`top_capture.txt`).
- Gathered screenshots of the Azure Portal pages for the resource group, storage account (showing the uploaded blob), and VM networking/NSG rules.

Observed remote request timings (examples):
- Run 1: 19.152031s
- Run 2: 19.342630s
- Run 3: 19.436745s

Artifacts captured:
- `response1.json`, `response2.json`, `response3.json` — JSON responses from the endpoint
- `top_capture.txt` — CPU/memory snapshot files collected on the VM during generation
- `deployment_screenshots/*.png` — Portal and terminal screenshots

Current state:
- Remote endpoint is functional and returning model generations.
- A set of verification artifacts and logs has been collected and stored locally for inclusion in the assignment deliverables.
- Step 9 (deeper performance measurement and analysis) is now in-progress.

### Step 9 - Collect performance observations

Status: Complete

What was done:
- Captured server-side runtime of the native binary using `/usr/bin/time -v` and saved the output to `llmtest_time.log`.

Observed sample timings:
- Remote HTTP end-to-end: ~19.15s, ~19.34s, ~19.44s
- Local (dev machine) end-to-end: ~6.10s (from earlier local tests)

Artifacts produced:
- `llmtest_time.log` — server-side `/usr/bin/time -v` output for a sample run

Current state:
- Performance data has been collected and saved as artifacts for analysis and for inclusion in the submission.
- Step 10 (package deliverables) is now in-progress.

## Current Project State

Right now the project is at the point where:
- The C++ inference wrapper works locally.
- The test executable accepts prompts and returns generated text.
- A small Flask API can call the executable and return JSON.
- The Flask endpoint has been tested locally and request timing has been captured.
- The Azure storage account and blob container have been created and the model has been uploaded.
- The next logical step is to create the Azure VM and prepare deployment.

## Notes for Later Reporting

Useful things to capture as the project continues:
- Exact build commands used.
- Exact prompt examples used for testing.
- Endpoint URL and sample JSON request/response.
- Any performance observations such as cold start time or generation latency.
- Screenshots of the cloud setup and service running.

## Update Rule

After each major step, add a short entry here with:
- status
- what changed
- current state
- any issues or observations
