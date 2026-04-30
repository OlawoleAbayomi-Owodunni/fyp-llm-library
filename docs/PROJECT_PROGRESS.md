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

## Current Project State

Right now the project is at the point where:
- The C++ inference wrapper works locally.
- The test executable accepts prompts and returns generated text.
- A small Flask API can call the executable and return JSON.
- The next logical step is to test the web service more carefully and then move toward deployment on a cloud VM.

## Next Planned Step

### Step 4 - Test the wrapped service locally

Planned work:
- Run the Flask app locally.
- Send a sample request to the `/generate` endpoint.
- Confirm the response format is correct.
- Record the request and response flow for later use in the report.

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
