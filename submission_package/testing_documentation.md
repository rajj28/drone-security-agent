# Testing Documentation

## Test Strategy

The system is validated through artifact-based tests and API checks that verify each stage of the pipeline.

## What Is Tested

- Frame extraction artifacts exist and extracted images are present.
- Telemetry JSON records contain timestamps, locations, zone flags, and drone context.
- Vision analysis records contain frame IDs, descriptions, threat values, and confidence values.
- Pinecone indexing logs match the analyzed frame count.
- Alert records contain severity, trigger status, and required fields.
- Session summary and QA logs are present and readable.
- API endpoints respond correctly using a TestClient-based test setup.
- End-to-end artifact consistency is maintained across telemetry, analysis, and alert outputs.

## Representative Scenarios

- A vehicle or person appearing in a frame should be logged with context.
- After-hours or restricted-zone presence should trigger a high-severity alert.
- Historical queries should retrieve matching object or activity events.
- Session summary should reflect cumulative monitoring statistics.

## Notes

- Tests are deterministic and based on generated artifacts.
- API tests use TestClient rather than a live localhost server.
- Duplicate frame processing is guarded through session context tracking.
